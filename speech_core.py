import base64
import io
import os
import re
import tempfile
import time
import unicodedata
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
WER_PUNCTUATION = re.compile(
    "[" + re.escape("।॥!?,.;:\"'`()[]{}-–—_/\\|<>*&^%$#@+=~") + "]"
)


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
    text = nemo_transcribe_segments(load_koyal(), [audio], sample_rate)[0]
    elapsed = time.perf_counter() - started
    if not text:
        raise RuntimeError("Koyal returned an empty transcript.")
    return text, elapsed


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
