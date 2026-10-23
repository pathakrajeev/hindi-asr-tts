import base64
import io
import os
import re
import tempfile
import time
import unicodedata
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import requests
import soundfile as sf
import streamlit as st
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from streamlit.errors import StreamlitSecretNotFoundError

load_dotenv(Path(__file__).resolve().parent / ".env")

st.set_page_config(page_title="Hindi Live ASR → TTS", page_icon="🗣️", layout="centered")

MODEL_ID = "adalat-ai/koyal-hi-120m-1.0"
MODEL_REVISION = "7f2cb579598ba607b7ebb78a1b523fbcb83f4cf9"
CHECKPOINT_FILE = "koyal-hi-120m-1.0.nemo"
SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 30
MAX_TTS_CHARS = 1_200
WER_PUNCTUATION = re.compile(
    "[" + re.escape("।॥!?,.;:\"'`()[]{}-–—_/\\|<>*&^%$#@+=~") + "]"
)


def secret(name: str) -> str:
    """Get secret from Streamlit secrets or environment variables. Never log the value."""
    try:
        value = st.secrets.get(name, "")
    except StreamlitSecretNotFoundError:
        value = ""
    return str(value or os.environ.get(name, "")).strip()


def dependency_message() -> str | None:
    if not secret("HF_TOKEN"):
        return "HF_TOKEN is missing. Add it to Streamlit secrets or the deployment environment."
    if not secret("SARVAM_API_KEY"):
        return "SARVAM_API_KEY is missing. Add it to Streamlit secrets or the deployment environment."
    return None


@st.cache_resource(show_spinner="Loading Koyal Hindi ASR model…")
def load_koyal():
    import nemo.collections.asr as nemo_asr

    checkpoint = hf_hub_download(
        MODEL_ID,
        CHECKPOINT_FILE,
        revision=MODEL_REVISION,
        token=secret("HF_TOKEN"),
    )
    model = nemo_asr.models.ASRModel.restore_from(checkpoint, map_location="cpu")
    model = model.half()
    model.eval()
    torch.cuda.empty_cache()
    return model


def decode_audio(raw: bytes, suffix: str) -> tuple[np.ndarray, int, float]:
    if not raw or len(raw) == 0:
        raise ValueError("Audio file is empty.")
    if len(raw) > 100 * 1024 * 1024:
        raise ValueError("Audio file is too large (max 100 MB).")

    with tempfile.NamedTemporaryFile(suffix=suffix or ".wav", delete=False) as file:
        file.write(raw)
        path = file.name
    try:
        audio, sample_rate = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise ValueError("Failed to decode audio. Ensure it's a valid audio file.") from exc
    finally:
        Path(path).unlink(missing_ok=True)

    audio = np.asarray(audio, dtype=np.float32)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("The uploaded audio is empty or invalid.")
    duration = len(audio) / sample_rate
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(f"Audio must be ≤{MAX_AUDIO_SECONDS}s; received {duration:.1f}s.")

    peak = float(np.max(np.abs(audio)))
    if peak:
        audio = np.clip(audio * min(1.0, 0.95 / peak), -1.0, 1.0)
    return audio, sample_rate, duration


def transcribe(audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
        path = file.name
    try:
        sf.write(path, audio, sample_rate, subtype="PCM_16")
        started = time.perf_counter()
        with torch.inference_mode():
            result = load_koyal().transcribe([path], batch_size=1)
        elapsed = time.perf_counter() - started
    finally:
        Path(path).unlink(missing_ok=True)
    if isinstance(result, tuple):
        result = result[0]
    value = result[0]
    text = value if isinstance(value, str) else getattr(value, "text", "")
    text = str(text).strip()
    if not text:
        raise RuntimeError("Koyal returned an empty transcript.")
    return text, elapsed


def synthesize(text: str) -> tuple[bytes, float, float]:
    text = " ".join(text.split())
    if len(text) > MAX_TTS_CHARS:
        raise ValueError("The transcript is too long for live TTS.")
    started = time.perf_counter()
    try:
        response = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": secret("SARVAM_API_KEY"), "Content-Type": "application/json"},
            json={
                "text": text,
                "language_code": "hi-IN",
                "speaker": "priya",
                "model": "bulbul:v3",
                "pace": 1.0,
                "speech_sample_rate": 24000,
                "output_audio_codec": "wav",
                "temperature": 0.6,
            },
            timeout=180,
        )
        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            if response.status_code == 401:
                raise RuntimeError("TTS authentication failed. Check your SARVAM_API_KEY.")
            else:
                raise RuntimeError(f"TTS service error ({response.status_code}). Please try again.")
        encoded = response.json().get("audios") or []
        if not encoded:
            raise RuntimeError("TTS service returned no audio. Please try again.")
        parts, output_rate = [], None
        for value in encoded:
            part, rate = sf.read(io.BytesIO(base64.b64decode(value)), dtype="float32", always_2d=False)
            part = part.mean(axis=1) if np.ndim(part) > 1 else part
            if output_rate is not None and output_rate != rate:
                raise RuntimeError("TTS returned inconsistent audio formats.")
            output_rate = int(rate)
            parts.append(np.asarray(part, dtype=np.float32))
        output = np.concatenate(parts)
        wav = io.BytesIO()
        sf.write(wav, output, output_rate, format="WAV", subtype="PCM_16")
        return wav.getvalue(), elapsed, len(output) / output_rate
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error during TTS. Please check your connection and try again.") from exc


def normalize_reference_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = "".join(
        str(unicodedata.decimal(character)) if character.isdecimal() else character
        for character in normalized
    )
    return " ".join(WER_PUNCTUATION.sub(" ", normalized).split())


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def score_transcript(reference_text: str, transcript: str) -> dict[str, float | int]:
    reference = normalize_reference_text(reference_text)
    hypothesis = normalize_reference_text(transcript)
    if not reference:
        raise ValueError("The golden-reference transcript is empty after normalization.")
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    reference_characters = reference.replace(" ", "")
    hypothesis_characters = hypothesis.replace(" ", "")
    return {
        "WER (%)": edit_distance(reference_words, hypothesis_words) / len(reference_words) * 100,
        "CER (%)": (
            edit_distance(reference_characters, hypothesis_characters)
            / len(reference_characters)
            * 100
        ),
        "Reference words": len(reference_words),
        "Reference characters": len(reference_characters),
    }


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


def render_research_analysis(
    input_audio: np.ndarray,
    input_rate: int,
    generated_wav: bytes,
    transcript: str,
    input_seconds: float,
    asr_seconds: float,
    tts_seconds: float,
    generated_seconds: float,
    reference_text: str,
) -> None:
    generated_audio, generated_rate = sf.read(
        io.BytesIO(generated_wav), dtype="float32", always_2d=False
    )
    if np.ndim(generated_audio) > 1:
        generated_audio = generated_audio.mean(axis=1)

    input_metrics = audio_summary(input_audio, input_rate, transcript)
    generated_metrics = audio_summary(generated_audio, int(generated_rate), transcript)
    asr_scores = score_transcript(reference_text, transcript)

    st.subheader("🔬 Research Analysis")
    st.caption("Detailed ASR/TTS evaluation with WER/CER metrics and audio analysis.")

    analysis_tabs = st.tabs(["ASR Accuracy", "Audio Metrics", "Spectrograms", "Pipeline Performance"])

    with analysis_tabs[0]:
        st.write(f"**Transcript**: {len(transcript.split())} words, {len(transcript)} characters")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            wer = asr_scores['WER (%)']
            st.metric("WER (%)", f"{wer:.2f}", delta=f"Reference: {asr_scores['Reference words']} words" if wer < 20 else "Higher error", delta_color="inverse")
        with col2:
            cer = asr_scores['CER (%)']
            st.metric("CER (%)", f"{cer:.2f}", delta=f"Ref: {asr_scores['Reference characters']} chars" if cer < 20 else "Higher error", delta_color="inverse")
        with col3:
            st.metric("Ref Words", asr_scores["Reference words"])
        with col4:
            st.metric("Ref Chars", asr_scores["Reference characters"])

    with analysis_tabs[1]:
        st.caption("Words per minute, spectral centroid, zero-crossing rate, and RMS/Peak levels.")
        metric_df = [
            {
                "Measure": measure,
                "Input": f"{input_value:.2f}",
                "Generated": f"{generated_metrics[measure]:.2f}",
                "Diff": f"{generated_metrics[measure] - input_value:+.2f}",
            }
            for measure, input_value in input_metrics.items()
        ]
        st.dataframe(metric_df, hide_index=True, use_container_width=True)

    with analysis_tabs[2]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        fig.patch.set_facecolor('white')

        for idx, (audio, sample_rate, label, waveform_axis, spectrum_axis) in enumerate((
            (input_audio, input_rate, "🎙️ Input Speech", axes[0, 0], axes[1, 0]),
            (generated_audio, int(generated_rate), "🔊 Generated Speech", axes[0, 1], axes[1, 1]),
        )):
            stride = max(1, int(np.ceil(len(audio) / 3000)))
            waveform = audio[::stride]
            waveform_axis.plot(np.arange(len(waveform)) * stride / sample_rate, waveform, linewidth=0.7, color='#1f77b4')
            waveform_axis.set(title=f"{label} Waveform", xlabel="Time (s)", ylabel="Amplitude")
            waveform_axis.grid(True, alpha=0.3)

            spectrum = np.abs(librosa.stft(audio, n_fft=512, hop_length=128, pad_mode="constant"))
            spectrum_db = librosa.amplitude_to_db(spectrum, ref=np.max)
            image = librosa.display.specshow(
                spectrum_db,
                sr=sample_rate,
                hop_length=128,
                x_axis="time",
                y_axis="hz",
                ax=spectrum_axis,
                cmap='viridis',
            )
            spectrum_axis.set(title=f"{label} Spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")
            fig.colorbar(image, ax=spectrum_axis, format="%+2.0f dB")

        st.pyplot(fig, use_container_width=True)

    with analysis_tabs[3]:
        st.markdown("### Pipeline Performance")
        perf_data = {
            "Stage": ["ASR (Koyal)", "TTS (Bulbul)"],
            "Processing Time (s)": [round(asr_seconds, 2), round(tts_seconds, 2)],
            "Audio Duration (s)": [round(input_seconds, 2), round(generated_seconds, 2)],
            "Real-Time Factor": [
                round(asr_seconds / input_seconds, 3) if input_seconds else 0.0,
                round(tts_seconds / generated_seconds, 3) if generated_seconds else 0.0,
            ],
        }
        st.dataframe(perf_data, hide_index=True, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            total_time = asr_seconds + tts_seconds
            total_audio = input_seconds + generated_seconds
            st.metric("Total Processing (s)", f"{total_time:.2f}")
        with col2:
            st.metric("Combined RTF", f"{(total_time / total_audio):.3f}" if total_audio > 0 else "N/A")


st.title("🗣️ Hindi Live ASR → TTS")
st.caption("Koyal Hindi ASR 120M → Sarvam Bulbul v3. Uploaded audio and reference text are not retained.")

tabs = st.tabs(["Live Processing", "Research Mode", "📚 Project Overview"])

with tabs[2]:
    st.markdown("""
    ## Developing ASR & TTS Systems in Hindi

    ### The Challenge
    Hindi is the **world's 3rd most spoken language** with 600M+ speakers, yet remains critically underserved by modern speech AI. Three challenges compound this:

    - **Phonetic Complexity**: Hindi has 45 phonemes (vs 24 in English). Retroflex ट vs dental त are phonemically distinct—changing meaning entirely. Example: ताल (rhythm) vs टाल (postpone).
    - **Data Scarcity**: Whisper trained on 680K hours English but only 12K hours Hindi—fundamentally under-prepared.
    - **Real-world Noise**: In railway stations and markets, WER exceeds 50%, making systems unusable.

    ### Additional Challenges
    - **Code-mixing (Hinglish)**: "Main office ja raha hoon, meeting hai at 3 pm"—monolingual models fail.
    - **Dialect Variation**: Bhojpuri, Rajasthani, Haryanvi introduce phonological variation.
    - **Synthesis Rules**: Schwa deletion (राम → /raam/, not /rama/) and question intonation require purpose-built design.

    ### Our Solution
    This system combines:
    - 🎯 **Koyal Hindi ASR 120M**: Fine-tuned Indic speech recognition
    - 🔊 **Sarvam Bulbul TTS v3**: Hindi-native text-to-speech synthesis
    - 📊 **Research Analysis**: WER/CER metrics, audio analysis, real-time performance
    """)

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Phonemes", "45", "vs 24 English (retroflex sounds)")
    with col2:
        st.metric("Training Data", "12K hrs", "vs 680K for English")
    with col3:
        st.metric("Real-world WER", ">50%", "Noisy environments")

with tabs[0]:
    st.subheader("Live Mode")
    st.info("Record or upload Hindi speech, then transcribe and synthesize in real-time.")
    mode = "Live"
    reference_text = ""
    reference_valid = True

with tabs[1]:
    st.subheader("Research Mode")
    st.info(
        "Upload a UTF-8 text file containing the verified transcript for this recording. "
        "Research mode calculates ASR accuracy (WER/CER) and detailed audio metrics."
    )
    mode = "Research"
    reference_text = ""
    reference_valid = True
    reference_file = st.file_uploader("Upload golden-reference transcript (.txt)", type=["txt"])
    if reference_file is None:
        st.warning("A golden-reference .txt file is required to run Research mode.")
        reference_valid = False
    else:
        try:
            reference_text = reference_file.getvalue().decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            st.error(f"The reference file must be UTF-8 text: {exc}")
            reference_valid = False
        if reference_valid and not normalize_reference_text(reference_text):
            st.warning("The reference file is empty or contains no scorable text.")
            reference_valid = False

with tabs[0]:
    st.divider()

problem = dependency_message()
if problem:
    st.error(problem)
    st.stop()

with tabs[0]:
    left, right = st.columns(2)
    with left:
        recording = st.audio_input("Record Hindi speech", key="live_recording")
    with right:
        upload = st.file_uploader("Or upload audio", type=["wav", "mp3", "m4a", "mp4", "flac", "ogg", "aac"], key="live_upload")

    source = recording or upload
    if source is not None:
        raw = source.getvalue()
        suffix = Path(getattr(source, "name", "recording.wav")).suffix or ".wav"
        st.audio(raw)
        if st.button("Transcribe and speak", type="primary", use_container_width=True, key="live_process"):
            try:
                with st.status("Processing live audio…", expanded=True) as status:
                    audio, sample_rate, input_seconds = decode_audio(raw, suffix)
                    st.write("Transcribing with Koyal…")
                    transcript, asr_seconds = transcribe(audio, sample_rate)
                    st.write("Generating speech with Bulbul…")
                    generated_wav, tts_seconds, generated_seconds = synthesize(transcript)
                    status.update(label="Complete", state="complete", expanded=False)
                st.subheader("📝 Transcript")
                st.write(transcript)
                st.subheader("🔊 Generated Speech")
                st.audio(generated_wav, format="audio/wav")
                st.divider()
                st.markdown("### ⚡ Performance Metrics")
                st.dataframe(
                    {
                        "Input seconds": [round(input_seconds, 2)],
                        "ASR seconds": [round(asr_seconds, 2)],
                        "ASR RTF": [round(asr_seconds / input_seconds, 3)],
                        "TTS seconds": [round(tts_seconds, 2)],
                        "TTS RTF": [round(tts_seconds / generated_seconds, 3)],
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Processing failed: {type(exc).__name__}: {exc}")
    else:
        st.info(f"🎤 Record or upload a Hindi utterance up to {MAX_AUDIO_SECONDS} seconds.")

with tabs[1]:
    left, right = st.columns(2)
    with left:
        recording = st.audio_input("Record Hindi speech", key="research_recording")
    with right:
        upload = st.file_uploader("Or upload audio", type=["wav", "mp3", "m4a", "mp4", "flac", "ogg", "aac"], key="research_upload")

    source = recording or upload
    if source is not None:
        raw = source.getvalue()
        suffix = Path(getattr(source, "name", "recording.wav")).suffix or ".wav"
        st.audio(raw)
        if st.button(
            "Transcribe and speak",
            type="primary",
            use_container_width=True,
            disabled=not reference_valid,
            key="research_process",
        ):
            try:
                with st.status("Processing research audio…", expanded=True) as status:
                    audio, sample_rate, input_seconds = decode_audio(raw, suffix)
                    st.write("Transcribing with Koyal…")
                    transcript, asr_seconds = transcribe(audio, sample_rate)
                    st.write("Generating speech with Bulbul…")
                    generated_wav, tts_seconds, generated_seconds = synthesize(transcript)
                    status.update(label="Complete", state="complete", expanded=False)
                st.subheader("📝 Transcript")
                st.write(transcript)
                st.subheader("🔊 Generated Speech")
                st.audio(generated_wav, format="audio/wav")
                st.divider()
                st.markdown("### ⚡ Performance Metrics")
                st.dataframe(
                    {
                        "Input seconds": [round(input_seconds, 2)],
                        "ASR seconds": [round(asr_seconds, 2)],
                        "ASR RTF": [round(asr_seconds / input_seconds, 3)],
                        "TTS seconds": [round(tts_seconds, 2)],
                        "TTS RTF": [round(tts_seconds / generated_seconds, 3)],
                    },
                    hide_index=True,
                    use_container_width=True,
                )
                render_research_analysis(
                    audio,
                    sample_rate,
                    generated_wav,
                    transcript,
                    input_seconds,
                    asr_seconds,
                    tts_seconds,
                    generated_seconds,
                    reference_text,
                )
            except Exception as exc:
                st.error(f"Processing failed: {type(exc).__name__}: {exc}")
    else:
        st.info(f"🎤 Record or upload a Hindi utterance up to {MAX_AUDIO_SECONDS} seconds.")
