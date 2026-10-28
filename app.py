import io
import logging
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import soundfile as sf
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

st.set_page_config(page_title="Hindi ASR → TTS Pipeline", page_icon="🗣️", layout="centered")

import hashlib  # noqa: E402

from research import render_research_tab  # noqa: E402
from batch_research import render_batch_tab  # noqa: E402
from deepfake import render_synthetic_mode  # noqa: E402
from speech_core import (
    EmptyTranscriptError,
    LIVE_ASR_WINDOW_SECONDS,
    audio_quality_preview,
    decode_audio,
    dependency_message,
    media_data_uri,
    normalize_reference_text,
    synthesize_in_chunks,
    transcribe,
)  # noqa: E402

MAX_AUDIO_SECONDS = 25 * 60
logger = logging.getLogger("hindi_asr_tts")


def render_figure(fig, alt: str) -> None:
    try:
        image = io.BytesIO()
        fig.savefig(image, format="png", bbox_inches="tight")
        st.image(media_data_uri(image.getvalue(), "image/png"), caption=alt, width="stretch")
    finally:
        plt.close(fig)


def audio_summary(audio: np.ndarray, sample_rate: int, text: str) -> dict[str, float | int]:
    duration = len(audio) / sample_rate
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if peak > 1e-8:
        active_intervals = librosa.effects.split(audio, top_db=40)
        active_seconds = sum(end - start for start, end in active_intervals) / sample_rate
    else:
        active_seconds = 0.0
    word_count = len(text.split())
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    zero_crossings = librosa.feature.zero_crossing_rate(y=audio)
    return {
        "Duration (s)": duration,
        "Words per minute": word_count / duration * 60 if duration else 0.0,
        "Non-silent audio (%)": float(active_seconds / duration * 100) if duration else 0.0,
        "RMS level (dBFS)": float(20 * np.log10(max(rms, 1e-8))),
        "Peak level (dBFS)": float(20 * np.log10(max(peak, 1e-8))),
        "Spectral centroid (Hz)": float(np.mean(centroid)),
        "Zero-crossing rate": float(np.mean(zero_crossings)),
    }


def render_metrics_dashboard(
    input_audio: np.ndarray,
    input_rate: int,
    generated_wav: bytes,
    transcript: str,
    input_seconds: float,
    asr_seconds: float,
    tts_seconds: float,
    generated_seconds: float,
) -> None:
    generated_audio, generated_rate = sf.read(io.BytesIO(generated_wav), dtype="float32", always_2d=False)
    if np.ndim(generated_audio) > 1:
        generated_audio = generated_audio.mean(axis=1)

    input_metrics = audio_summary(input_audio, input_rate, transcript)
    generated_metrics = audio_summary(generated_audio, int(generated_rate), transcript)

    st.subheader("📊 Speech metrics and charts")
    st.caption("Live audio, speech, and pipeline measurements. Reference-based ASR scores are in Research mode.")
    analysis_tabs = st.tabs(["Audio Metrics", "Audio Diagnostics", "Prosody", "Pipeline Performance"])

    with analysis_tabs[0]:
        st.caption("Words per minute, spectral centroid, zero-crossing rate, and RMS/Peak levels. "
                   "The input is peak-limited to 0.95 (about −0.45 dBFS) when decoded, so its peak and RMS "
                   "describe the scaled signal, not the original recording level.")
        st.dataframe(
            [
                {
                    "Measure": measure,
                    "Input": f"{input_value:.2f}",
                    "Generated": f"{generated_metrics[measure]:.2f}",
                    "Diff": f"{generated_metrics[measure] - input_value:+.2f}",
                }
                for measure, input_value in input_metrics.items()
            ],
            hide_index=True,
            width="stretch",
        )

    with analysis_tabs[1]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        signals = (
            (input_audio, input_rate, "Input speech", axes[0, 0], axes[1, 0], "#2a78d6"),
            (generated_audio, int(generated_rate), "Generated speech", axes[0, 1], axes[1, 1], "#eb6834"),
        )
        for audio, sample_rate, label, waveform_axis, spectrum_axis, color in signals:
            stride = max(1, int(np.ceil(len(audio) / 5000)))
            waveform = audio[::stride]
            waveform_axis.plot(np.arange(len(waveform)) * stride / sample_rate, waveform, linewidth=0.7, color=color)
            waveform_axis.set(title=f"{label} waveform", xlabel="Time (s)", ylabel="Amplitude")
            waveform_axis.grid(True, alpha=0.3)
            mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=512, hop_length=160, n_mels=64)
            image = librosa.display.specshow(
                librosa.power_to_db(mel, ref=np.max), sr=sample_rate, hop_length=160,
                x_axis="time", y_axis="mel", ax=spectrum_axis, cmap="viridis",
            )
            spectrum_axis.set(title=f"{label} log-Mel spectrogram", xlabel="Time (s)", ylabel="Mel frequency")
            fig.colorbar(image, ax=spectrum_axis, format="%+2.0f dB")
        render_figure(fig, "Input and generated speech waveforms and log-Mel spectrograms")

    with analysis_tabs[2]:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
        for audio, sample_rate, label, color in (
            (input_audio, input_rate, "Input speech", "#2a78d6"),
            (generated_audio, int(generated_rate), "Generated speech", "#eb6834"),
        ):
            hop_length = 256
            f0 = librosa.yin(audio, fmin=65, fmax=min(500, sample_rate / 2 - 1), sr=sample_rate,
                             frame_length=1024, hop_length=hop_length)
            rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=hop_length)[0]
            frame_count = min(len(f0), len(rms))
            times = librosa.frames_to_time(np.arange(frame_count), sr=sample_rate, hop_length=hop_length)
            voiced_f0 = np.where(rms[:frame_count] > max(float(np.max(rms)) * 0.02, 1e-5), f0[:frame_count], np.nan)
            axes[0].plot(times, voiced_f0, linewidth=0.8, label=label, color=color)
            axes[1].plot(times, 20 * np.log10(np.maximum(rms[:frame_count], 1e-8)), linewidth=0.8, label=label, color=color)
        axes[0].set(title="Estimated pitch contour", ylabel="Pitch (Hz)")
        axes[1].set(title="Frame energy contour", xlabel="Time (s)", ylabel="RMS (dBFS)")
        for axis in axes:
            axis.grid(True, alpha=0.25)
            axis.legend()
        st.caption("Pitch is estimated with YIN and masked during low-energy frames; it is descriptive, not a voice-quality score.")
        render_figure(fig, "Estimated pitch and RMS energy contours for the input and generated speech")

    with analysis_tabs[3]:
        asr_rtf = asr_seconds / input_seconds if input_seconds else 0.0
        tts_rtf = tts_seconds / generated_seconds if generated_seconds else 0.0
        st.caption("ASR time is Koyal inference only (model loading excluded). TTS time is the Bulbul request time. "
                   "Decoding and chart rendering are not included.")
        columns = st.columns(3)
        columns[0].metric("ASR + TTS processing (s)", f"{asr_seconds + tts_seconds:.2f}")
        columns[1].metric("ASR real-time factor", f"{asr_rtf:.3f}", help="ASR processing time divided by input audio duration; lower is faster.")
        columns[2].metric("TTS real-time factor", f"{tts_rtf:.3f}", help="TTS processing time divided by generated audio duration; lower is faster.")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].barh(["ASR (Koyal)", "TTS (Bulbul)"], [asr_seconds, tts_seconds], color=["#2a78d6", "#eb6834"])
        axes[0].set(title="Processing latency", xlabel="Seconds")
        axes[1].barh(["ASR (Koyal)", "TTS (Bulbul)"], [asr_rtf, tts_rtf], color=["#2a78d6", "#eb6834"])
        axes[1].axvline(1, color="gray", linestyle="--", linewidth=1, label="Real-time")
        axes[1].set(title="Real-time factor", xlabel="Processing time / audio duration")
        axes[1].legend()
        for axis in axes:
            axis.grid(axis="x", alpha=0.25)
        render_figure(
            fig,
            "Horizontal bars compare ASR and TTS latency and real-time factor; lower values are faster",
        )


def render_live_tab() -> None:
    st.subheader("Live mode")
    st.info("Record or upload Hindi speech, review and correct Koyal's transcript, then generate speech from the corrected text.")
    left, right = st.columns(2)
    with left:
        recording = st.audio_input("Record Hindi speech", key="live_recording")
    with right:
        upload = st.file_uploader("Or upload audio", type=["wav", "mp3", "m4a", "mp4", "flac", "ogg", "aac"], key="live_upload")
    source = recording or upload
    quality = None
    if source is None:
        st.info("🎤 Record or upload Hindi audio up to 25 minutes (maximum file size: 50 MB).")
    else:
        raw = source.getvalue()
        source_name = getattr(source, "name", "recording.wav")
        size_mb = len(raw) / (1024 * 1024)
        st.caption(f"Selected: {source_name} · {size_mb:.1f} MB")
        try:
            quality = audio_quality_preview(raw, Path(source_name).suffix, MAX_AUDIO_SECONDS)
            st.caption(
                f"Audio check · {quality['duration']:.1f} s · "
                f"peak {quality['peak']:.3f} · RMS {quality['rms_dbfs']:.1f} dBFS"
            )
            if quality["rms_dbfs"] < -35:
                st.warning("This recording is very quiet; try speaking closer to the microphone before transcription.")
            if quality["clipping_fraction"] >= 0.005:
                st.warning("This recording may be clipped; reduce the microphone input level before transcription.")
        except Exception as exc:
            st.error(f"Could not inspect the recording: {type(exc).__name__}: {exc}")
    source_digest = hashlib.sha256(source.getvalue()).hexdigest() if source is not None else None
    stored = st.session_state.get("live_result")
    if stored is not None and stored["digest"] != source_digest:
        st.session_state.pop("live_result", None)  # new or removed audio: old result no longer applies
        stored = None
    if st.button(
        "Transcribe audio",
        type="primary",
        width="stretch",
        disabled=source is None or quality is None,
        key="live_process",
    ):
        if source is None:
            st.error("Record or upload audio before starting transcription.")
            return
        stored = _process_live(source, source_digest)
    if stored is not None:
        _render_live_result(stored)


def _process_live(source, source_digest):
    raw = source.getvalue()
    try:
        with st.status("Processing live audio…", expanded=True) as status:
            audio, sample_rate, input_seconds, audio_info = decode_audio(
                raw, Path(getattr(source, "name", "recording.wav")).suffix, MAX_AUDIO_SECONDS,
                return_info=True,
            )
            if input_seconds > LIVE_ASR_WINDOW_SECONDS:
                st.write(
                    f"Transcribing {input_seconds / 60:.1f} minutes with Koyal "
                    f"in speech-aware windows of up to {LIVE_ASR_WINDOW_SECONDS} s…"
                )
            else:
                st.write("Transcribing with Koyal…")
            transcript, asr_seconds = transcribe(audio, sample_rate)
            status.update(label="Transcription complete", state="complete", expanded=False)
        result = {
            "digest": source_digest, "audio": audio, "sample_rate": sample_rate, "input_seconds": input_seconds,
            "audio_info": audio_info, "transcript": transcript, "asr_seconds": asr_seconds,
        }
        st.session_state["live_result"] = result
        return result
    except EmptyTranscriptError as exc:
        logger.warning("Live ASR produced no transcript: %s", exc.diagnostics)
        st.error("Koyal did not recognize any words in this recording.")
        st.caption(exc.diagnostics)
        st.info(
            "Try a clear 5–10 second Hindi recording with speech close to the microphone, "
            "or upload a WAV recording. If it still returns no words, compare with Sarvam Saaras "
            "in Research mode to check whether the issue is specific to Koyal."
        )
    except Exception as exc:
        logger.exception("Live speech processing failed")
        st.error(f"Processing failed: {type(exc).__name__}: {exc}")
    return None


def _render_live_result(result) -> None:
    st.subheader("Review and correct the transcript")
    edited_text = st.text_area(
        "Transcript to synthesize",
        value=result["transcript"],
        height=120,
        key=f"live_transcript_{result['digest']}",
        help="Correct recognition mistakes here. Speech generation uses this edited text.",
    ).strip()
    if result.get("generated_text") != edited_text:
        result.pop("generated_wav", None)
        result.pop("generated_seconds", None)
        result.pop("tts_seconds", None)
        result.pop("tts_chunks", None)
    if st.button(
        "Generate speech from reviewed transcript",
        type="primary",
        width="stretch",
        disabled=not bool(normalize_reference_text(edited_text)),
        key=f"live_synthesize_{result['digest']}",
    ):
        try:
            with st.status("Generating speech with Bulbul…", expanded=True) as status:
                generated_wav, tts_seconds, generated_seconds, tts_chunks = synthesize_in_chunks(
                    edited_text,
                    on_chunk=lambda index, total: status.write(
                        f"Generating speech chunk {index} of {total}…"
                    ),
                )
                status.update(label="Speech generation complete", state="complete", expanded=False)
            result.update({
                "generated_text": edited_text,
                "generated_wav": generated_wav,
                "tts_seconds": tts_seconds,
                "generated_seconds": generated_seconds,
                "tts_chunks": tts_chunks,
            })
            st.session_state["live_result"] = result
        except Exception as exc:
            logger.exception("Live TTS failed")
            st.error(f"Speech generation failed: {type(exc).__name__}: {exc}")

    st.subheader("Transcription diagnostics")
    peak = result["audio_info"]["input_peak"]
    rms = float(np.sqrt(np.mean(np.square(result["audio"], dtype=np.float64)))) if len(result["audio"]) else 0.0
    rms_dbfs = float(20 * np.log10(max(rms, 1e-8)))
    st.caption(f"Input duration: {result['input_seconds']:.1f} s · Peak: {peak:.3f} · RMS: {rms_dbfs:.1f} dBFS")
    if rms_dbfs < -35:
        st.warning("The recording is very quiet. Try speaking closer to the microphone and record again.")
    if result["audio_info"]["clipping_fraction"] >= 0.005:
        st.warning("The recording may be clipped. Reduce the microphone input level and try again.")

    if "generated_wav" in result and result.get("generated_text") == edited_text:
        if result["tts_chunks"] > 1:
            st.caption(f"Speech was generated in {result['tts_chunks']} chunks to fit the TTS service limit.")
        st.subheader("Generated speech")
        st.audio(media_data_uri(result["generated_wav"], "audio/wav"), format="audio/wav")
    st.divider()
    if "generated_wav" in result and result.get("generated_text") == edited_text:
        try:
            render_metrics_dashboard(
                result["audio"], result["sample_rate"], result["generated_wav"], edited_text,
                result["input_seconds"], result["asr_seconds"], result["tts_seconds"], result["generated_seconds"],
            )
        except Exception:
            logger.exception("Live speech metrics rendering failed")
            st.warning("Speech generation succeeded, but its metrics and charts could not be displayed.")


def render_mode_overview(mode: str) -> None:
    with st.expander(f"Project overview · {mode}", expanded=False):
        st.markdown("""
        ## Hindi speech recognition and synthesis

        Hindi ASR and TTS need to handle phonetic distinctions, regional pronunciation,
        background noise, and Hindi-English code-mixing. This app provides a fast
        single-model workflow and a configurable evaluation workflow.
        """)
        if mode == "Live":
            st.markdown("""
            **Live pipeline**
            1. Record or upload a Hindi audio clip (up to 25 minutes; maximum file size 50 MB).
            2. Transcribe it with Koyal Hindi ASR 120M and review/correct the editable transcript.
            3. Generate speech with Sarvam Bulbul v3 only after you confirm the text.
            4. Review the generated audio, recording diagnostics, pitch and energy contours,
               processing latency, and real-time factors.

            Live mode does not require a reference transcript, so it does not report
            reference-based WER or CER.
            """)
        elif mode == "Synthetic speech":
            st.markdown("""
            **Synthetic speech (deepfake) detection**
            1. *Check a file*: score any clip with a zero-shot Indic detector and, once trained, your own detector.
            2. *Build dataset*: add the last Batch Research run (inputs = real, TTS outputs = synthetic) and any
               labelled files; every clip gets the same loudness and MP3 codec treatment.
            3. *Train & evaluate*: WavLM embeddings + logistic regression, tested on unseen speakers and on
               TTS engines left out of training (AUC, EER, Brier).

            Scores are research evidence, not proof that a recording is genuine or fake.
            """)
        elif mode in ("Research", "Batch Research"):
            st.markdown("""
            **Research pipeline**
            1. Provide a golden-reference transcript and record or upload audio
               (up to 120 seconds).
            2. Choose the language variety (Hindi, Hindi-English code-mixed, Sanskrit) and the
               ASR and TTS models to compare on the same recording. Unsupported model-language
               pairs are reported, not run.
            3. Use the reference, an ASR transcript, or reviewed custom text as TTS input.
               Maya 2 Native is available through a configured remote speech API for Hindi.
            4. Review preprocessing and VAD diagnostics, normalized / raw / script-tolerant WER,
               CER and word-level alignment, inference latency and real-time factor (model
               loading shown separately), TTS loopback and optional UTMOS scores, voice control,
               prosody and voice measurements, and an optional synthetic-speech check.
            5. Batch Research adds corpus WER/CER with 95% confidence intervals per variety.

            Local models load only when selected. Model availability and memory
            requirements vary by deployment.
            """)


st.title("🗣️ Hindi ASR → TTS Pipeline")
mode = st.radio(
    "Choose a workflow",
    ["Research", "Live", "Batch Research", "Synthetic speech"],
    horizontal=True,
    key="app_mode",
)
render_mode_overview(mode)

# Live always uses Koyal (gated, HF_TOKEN) and Bulbul (SARVAM_API_KEY). The other modes warn
# about missing secrets only for the models actually selected.
if mode == "Live":
    problem = dependency_message(("HF_TOKEN", "SARVAM_API_KEY"))
    if problem:
        st.error(problem)
        st.stop()

if mode == "Live":
    render_live_tab()
elif mode == "Research":
    render_research_tab()
elif mode == "Batch Research":
    render_batch_tab()
else:
    render_synthetic_mode()
