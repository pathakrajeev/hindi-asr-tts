import base64
import io
import os
import re
import tempfile
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from streamlit.errors import StreamlitSecretNotFoundError

KOYAL_MODEL_ID = "adalat-ai/koyal-hi-120m-1.0"
KOYAL_REVISION = "7f2cb579598ba607b7ebb78a1b523fbcb83f4cf9"
KOYAL_CHECKPOINT = "koyal-hi-120m-1.0.nemo"
SAMPLE_RATE = 16_000
BULBUL_MAX_CHARS = 2_500
MAX_AUDIO_FILE_BYTES = 50 * 1024 * 1024
LIVE_ASR_WINDOW_SECONDS = 28
WER_PUNCTUATION = re.compile(
    "[" + re.escape("।॥!?,.;:\"'`()[]{}-–—_/\\|<>*&^%$#@+=~") + "]"
)


class EmptyTranscriptError(RuntimeError):
    def __init__(self, audio: np.ndarray, sample_rate: int):
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0
        active_threshold = max(peak * 0.02, 1e-4)
        active_percent = float(np.mean(np.abs(audio) > active_threshold) * 100) if len(audio) else 0.0
        duration = len(audio) / sample_rate if sample_rate else 0.0
        self.diagnostics = (
            f"Decoded audio: {duration:.1f}s at {sample_rate} Hz; "
            f"peak {peak:.3f}; RMS {20 * np.log10(max(rms, 1e-8)):.1f} dBFS; "
            f"{active_percent:.1f}% of samples above 2% of peak."
        )
        super().__init__("Koyal returned an empty transcript.")


def secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except StreamlitSecretNotFoundError:
        value = ""
    return str(value or os.environ.get(name, "")).strip()


def dependency_message() -> str | None:
    for name in ("HF_TOKEN", "SARVAM_API_KEY"):
        if not secret(name):
            return f"{name} is missing. Add it to Streamlit secrets or the deployment environment."
    return None


@st.cache_resource(show_spinner="Loading Koyal Hindi ASR model…")
def load_koyal():
    import nemo.collections.asr as nemo_asr

    checkpoint = hf_hub_download(
        KOYAL_MODEL_ID, KOYAL_CHECKPOINT, revision=KOYAL_REVISION, token=secret("HF_TOKEN")
    )
    return nemo_asr.models.ASRModel.restore_from(checkpoint, map_location="cpu").eval()


def decode_audio(raw: bytes, suffix: str, max_seconds: float) -> tuple[np.ndarray, int, float]:
    if not raw:
        raise ValueError("Audio file is empty.")
    if len(raw) > MAX_AUDIO_FILE_BYTES:
        raise ValueError("Audio file is too large (max 50 MB).")
    with tempfile.NamedTemporaryFile(suffix=suffix or ".wav", delete=False) as file:
        file.write(raw)
        path = file.name
    try:
        try:
            duration = librosa.get_duration(path=path)
        except Exception as exc:
            raise ValueError("Failed to inspect audio. Ensure it's a valid audio file.") from exc
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Audio file has no readable duration.")
        if duration > max_seconds:
            raise ValueError(f"Audio must be ≤{max_seconds:.0f}s; received {duration:.1f}s.")
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
    if duration > max_seconds:
        raise ValueError(f"Audio must be ≤{max_seconds:.0f}s; received {duration:.1f}s.")
    peak = float(np.max(np.abs(audio)))
    if peak:
        audio = np.clip(audio * min(1.0, 0.95 / peak), -1.0, 1.0)
    return audio, sample_rate, duration


def _hypothesis_text(value) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    text = value if isinstance(value, str) else getattr(value, "text", "")
    return str(text or "").strip()


def nemo_transcribe_segments(model, segments: list[np.ndarray], sample_rate: int) -> list[str]:
    paths = []
    try:
        for segment in segments:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
                paths.append(file.name)
            sf.write(paths[-1], segment, sample_rate, subtype="PCM_16")
        with torch.inference_mode():
            result = model.transcribe(paths, batch_size=1)
        if isinstance(result, tuple):
            result = result[0]
        return [_hypothesis_text(value) for value in result]
    finally:
        for path in paths:
            Path(path).unlink(missing_ok=True)


def transcribe(audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
    started = time.perf_counter()
    window_samples = LIVE_ASR_WINDOW_SECONDS * sample_rate
    segments = (
        [audio[start:start + window_samples] for start in range(0, len(audio), window_samples)]
        if len(audio) > window_samples
        else [audio]
    )
    text = " ".join(nemo_transcribe_segments(load_koyal(), segments, sample_rate)).strip()
    elapsed = time.perf_counter() - started
    if not text:
        raise EmptyTranscriptError(audio, sample_rate)
    return text, elapsed


def split_tts_text(text: str, max_chars: int = BULBUL_MAX_CHARS) -> list[str]:
    text = " ".join(text.split())
    if not text:
        raise ValueError("There is no transcript text to synthesize.")
    if max_chars < 1:
        raise ValueError("The TTS chunk size must be positive.")

    chunks, current = [], ""
    sentences = re.split(r"(?<=[.!?।॥])\s+", text)
    for sentence in sentences:
        if len(sentence) <= max_chars:
            parts = [sentence]
        else:
            parts, part = [], ""
            for word in sentence.split():
                if len(word) > max_chars:
                    raise ValueError(
                        f"A transcript word exceeds Bulbul's {max_chars}-character limit."
                    )
                if part and len(part) + 1 + len(word) > max_chars:
                    parts.append(part)
                    part = ""
                part = f"{part} {word}".strip()
            if part:
                parts.append(part)

        for part in parts:
            if current and len(current) + 1 + len(part) > max_chars:
                chunks.append(current)
                current = ""
            current = f"{current} {part}".strip()
    if current:
        chunks.append(current)
    return chunks


def synthesize(text: str) -> tuple[bytes, float, float]:
    text = " ".join(text.split())
    if len(text) > BULBUL_MAX_CHARS:
        raise ValueError(f"Bulbul accepts up to {BULBUL_MAX_CHARS} characters.")
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
    except requests.RequestException as exc:
        raise RuntimeError("Network error during TTS. Check your connection and try again.") from exc
    elapsed = time.perf_counter() - started
    if response.status_code == 401:
        raise RuntimeError("TTS authentication failed. Check your SARVAM_API_KEY.")
    if response.status_code != 200:
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


def synthesize_in_chunks(
    text: str,
    on_chunk: Callable[[int, int], None] | None = None,
) -> tuple[bytes, float, float, int]:
    chunks = split_tts_text(text)
    wav = io.BytesIO()
    writer = None
    output_rate = None
    elapsed, duration = 0.0, 0.0

    try:
        for index, chunk in enumerate(chunks, start=1):
            if on_chunk is not None:
                on_chunk(index, len(chunks))
            wav_bytes, chunk_elapsed, chunk_duration = synthesize(chunk)
            audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
            audio = audio.mean(axis=1) if np.ndim(audio) > 1 else audio
            if output_rate is not None and output_rate != sample_rate:
                raise RuntimeError("Bulbul returned inconsistent sample rates between text chunks.")
            output_rate = int(sample_rate)
            if writer is None:
                writer = sf.SoundFile(
                    wav, mode="w", samplerate=output_rate, channels=1, format="WAV", subtype="PCM_16"
                )
            writer.write(np.asarray(audio, dtype=np.float32))
            elapsed += chunk_elapsed
            duration += chunk_duration
    finally:
        if writer is not None:
            writer.close()

    return wav.getvalue(), elapsed, duration, len(chunks)


def normalize_reference_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = "".join(
        str(unicodedata.decimal(character)) if character.isdecimal() else character
        for character in normalized
    )
    return " ".join(WER_PUNCTUATION.sub(" ", normalized).split())


def edit_distance(reference, hypothesis) -> int:
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


def align_words(reference: list[str], hypothesis: list[str]) -> list[tuple[str, str, str]]:
    """Levenshtein alignment as (op, ref_word, hyp_word); op is ok/sub/del/ins."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = np.zeros((rows, cols), dtype=np.int32)
    cost[:, 0] = np.arange(rows)
    cost[0, :] = np.arange(cols)
    for i in range(1, rows):
        for j in range(1, cols):
            cost[i, j] = min(
                cost[i - 1, j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                cost[i - 1, j] + 1,
                cost[i, j - 1] + 1,
            )
    ops, i, j = [], rows - 1, cols - 1
    while i or j:
        if i and j and cost[i, j] == cost[i - 1, j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            ops.append(("ok" if reference[i - 1] == hypothesis[j - 1] else "sub", reference[i - 1], hypothesis[j - 1]))
            i, j = i - 1, j - 1
        elif i and cost[i, j] == cost[i - 1, j] + 1:
            ops.append(("del", reference[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hypothesis[j - 1]))
            j -= 1
    return ops[::-1]


def score_transcript(reference_text: str, transcript: str) -> dict:
    reference = normalize_reference_text(reference_text)
    hypothesis = normalize_reference_text(transcript)
    if not reference:
        raise ValueError("The reference transcript is empty after normalization.")
    ref_words, hyp_words = reference.split(), hypothesis.split()
    ops = align_words(ref_words, hyp_words)
    ref_chars, hyp_chars = reference.replace(" ", ""), hypothesis.replace(" ", "")
    return {
        "wer": edit_distance(ref_words, hyp_words) / len(ref_words) * 100,
        "cer": edit_distance(ref_chars, hyp_chars) / len(ref_chars) * 100,
        "substitutions": sum(op == "sub" for op, _, _ in ops),
        "deletions": sum(op == "del" for op, _, _ in ops),
        "insertions": sum(op == "ins" for op, _, _ in ops),
        "reference_words": len(ref_words),
        "alignment": ops,
    }
