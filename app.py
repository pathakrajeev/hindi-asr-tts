import io
import logging
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

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

from research import render_research_tab  # noqa: E402
from speech_core import (
    EmptyTranscriptError,
    LIVE_ASR_WINDOW_SECONDS,
    decode_audio,
    dependency_message,
    media_data_uri,
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
        st.caption("Words per minute, spectral centroid, zero-crossing rate, and RMS/Peak levels.")
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
        columns = st.columns(3)
        columns[0].metric("Total processing (s)", f"{asr_seconds + tts_seconds:.2f}")
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
    st.info("Record or upload Hindi speech, then transcribe with Koyal and speak it back with Bulbul.")
    left, right = st.columns(2)
    with left:
        recording = st.audio_input("Record Hindi speech", key="live_recording")
    with right:
        upload = st.file_uploader("Or upload audio", type=["wav", "mp3", "m4a", "mp4", "flac", "ogg", "aac"], key="live_upload")
    source = recording or upload
    if source is None:
        st.info("🎤 Record or upload Hindi audio up to 25 minutes (maximum file size: 50 MB).")
    else:
        raw = source.getvalue()
        source_name = getattr(source, "name", "recording.wav")
        size_mb = len(raw) / (1024 * 1024)
        st.caption(f"Selected: {source_name} · {size_mb:.1f} MB")
    if not st.button(
        "Transcribe and speak",
        type="primary",
        width="stretch",
        key="live_process",
    ):
        return
    if source is None:
        st.error("Record or upload audio before starting transcription.")
        return
    raw = source.getvalue()
    try:
        with st.status("Processing live audio…", expanded=True) as status:
            audio, sample_rate, input_seconds = decode_audio(
                raw, Path(getattr(source, "name", "recording.wav")).suffix, MAX_AUDIO_SECONDS
            )
            if input_seconds > LIVE_ASR_WINDOW_SECONDS:
                segment_count = int(np.ceil(input_seconds / LIVE_ASR_WINDOW_SECONDS))
                st.write(
                    f"Transcribing {input_seconds / 60:.1f} minutes with Koyal "
                    f"in {segment_count} segments…"
                )
            else:
                st.write("Transcribing with Koyal…")
            transcript, asr_seconds = transcribe(audio, sample_rate)
            st.write("Generating speech with Bulbul…")
            generated_wav, tts_seconds, generated_seconds, tts_chunks = synthesize_in_chunks(
                transcript,
                on_chunk=lambda index, total: status.write(
                    f"Generating speech chunk {index} of {total}…"
                ),
            )
            status.update(label="Complete", state="complete", expanded=False)
        st.subheader("📝 Transcript")
        st.write(transcript)
        if tts_chunks > 1:
            st.caption(f"Speech was generated in {tts_chunks} chunks to fit the TTS service limit.")
        st.subheader("🔊 Generated speech")
        st.audio(media_data_uri(generated_wav, "audio/wav"), format="audio/wav")
        st.divider()
        try:
            render_metrics_dashboard(
                audio, sample_rate, generated_wav, transcript,
                input_seconds, asr_seconds, tts_seconds, generated_seconds,
            )
        except Exception:
            logger.exception("Live speech metrics rendering failed")
            st.warning("Speech generation succeeded, but its metrics and charts could not be displayed.")
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
            2. Transcribe it with Koyal Hindi ASR 120M.
            3. Generate speech from the transcript with Sarvam Bulbul v3.
            4. Review the transcript, generated audio, audio diagnostics, pitch and
               energy contours, processing latency, and real-time factors.

            Live mode does not require a reference transcript, so it does not report
            reference-based WER or CER.
            """)
        else:
            st.markdown("""
            **Research pipeline**
            1. Provide a golden-reference transcript and record or upload audio
               (up to 120 seconds).
            2. Select ASR and TTS models to compare on the same recording.
            3. Review preprocessing and VAD diagnostics, WER/CER and word-level
               alignment, model latency and real-time factor, TTS loopback and
               optional UTMOS scores, plus prosody and voice measurements.

            Local models load only when selected. Model availability and memory
            requirements vary by deployment.
            """)


st.title("🗣️ Hindi ASR → TTS Pipeline")
mode = st.radio(
    "Choose a workflow",
    ["Live", "Research"],
    horizontal=True,
    key="app_mode",
)
render_mode_overview(mode)

problem = dependency_message()
if problem:
    st.error(problem)
    st.stop()

if mode == "Live":
    render_live_tab()
else:
    render_research_tab()
