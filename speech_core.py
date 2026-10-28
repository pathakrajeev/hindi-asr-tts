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
CLIP_LEVEL = 0.999
ZERO_WIDTH = re.compile("[​‌‍﻿]")

# Language varieties the app understands. "hi-en" = Hindi-English code-mixed speech.
LANGUAGES = {"hi": "Hindi", "hi-en": "Hindi-English (code-mixed)", "sa": "Sanskrit"}
# Headline metric per variety, as in the notebook's Section 12B.
PRIMARY_METRIC = {"hi": "wer", "hi-en": "wer_script_tolerant", "sa": "cer"}
PRIMARY_LABEL = {"wer": "Normalized WER", "wer_script_tolerant": "Script-tolerant WER", "cer": "Normalized CER"}


# ---------------------------------------------------------------- language suggestion from text
# Common English words as Hindi speakers write them in Devanagari. A suggestion aid only:
# nativized words (स्कूल, बस, फोन) are common in plain Hindi, so code-mixing needs several of them.
ENGLISH_IN_DEVANAGARI = set("""
टीम मीटिंग मैनेजर प्रोजेक्ट डेडलाइन नेक्स्ट लास्ट फ्राइडे मंडे संडे सैटरडे ट्यूज़डे वेडनेसडे थर्सडे वीकेंड वीक मंथ
प्लान प्लानिंग वाइट व्हाइट ब्लैक बोर्ड व्हाइटबोर्ड टास्क टास्क्स डिवाइड फाइनल रिपोर्ट ईमेल मेल मैसेज कॉल कॉन्फ्रेंस
ऑफिस बॉस क्लाइंट कस्टमर सर्विस टारगेट सेल्स मार्केटिंग बजट अप्रूवल अप्रूव रिव्यू फीडबैक अपडेट स्टेटस इश्यू प्रॉब्लम
सॉल्यूशन सॉल्व प्रेजेंटेशन स्लाइड स्लाइड्स डॉक्यूमेंट फाइल फोल्डर शेयर सेंड फॉरवर्ड रिप्लाई चेक कन्फर्म कैंसिल
पोस्टपोन शेड्यूल टाइम लेट अर्ली सॉरी थैंक्स थैंक्यू प्लीज़ ओके हेलो हाय बाय गुड बेस्ट नाइस ग्रेट कूल सुपर
सिस्टम सर्वर डेटा डेटाबेस सॉफ्टवेयर हार्डवेयर ऐप एप्लिकेशन वेबसाइट लिंक पासवर्ड लॉगिन अकाउंट नेटवर्क इंटरनेट
वाईफाई ब्राउज़र डाउनलोड अपलोड इंस्टॉल कोड कोडिंग बग फिक्स टेस्ट टेस्टिंग रिलीज़ वर्ज़न फीचर डिज़ाइन यूज़र
कंप्यूटर लैपटॉप मोबाइल फोन चार्जर बैटरी स्क्रीन कीबोर्ड प्रिंटर स्कैन ऑनलाइन ऑफलाइन वीडियो ऑडियो फोटो
मैडम सर डॉक्टर नर्स हॉस्पिटल मेडिकल टिकट स्टेशन ट्रेन बस कैब टैक्सी फ्लाइट एयरपोर्ट होटल बुकिंग रूम
पार्टी बर्थडे गिफ्ट शॉपिंग मॉल मूवी शो प्रोग्राम मैच गेम टीचर स्टूडेंट क्लास एग्ज़ाम रिज़ल्ट सब्जेक्ट होमवर्क
कॉलेज यूनिवर्सिटी कोर्स लेक्चर असाइनमेंट सबमिट डेडलाइंस प्रोफेसर लैब रिसर्च पेपर जॉब इंटरव्यू सैलरी
""".split())
_HINDI_FUNCTION_WORDS = set("""
है हैं था थी थे हो होगा होगी का की के में से को और ने पर भी तो कि जो यह वह ये वो नहीं ही तक लिए कर रहा रही रहे
एक हम मैं आप तुम उस इस उन इन कुछ बहुत क्या कब कहाँ क्यों कैसे अब फिर गया गई गए जा जाता जाती
""".split())
_SANSKRIT_ENDINGS = re.compile(r"(?:ः|्|स्य|ानि|ेषु|भ्यः|त्वा|ात्|ाय|न्ति|ति|ते|म्)$")


def _deva_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[ऀ-ॿ]+", text or "") if re.search(r"[अ-हक़-ॡ]", w)]


def _loanword_key(word: str) -> str:
    import textnorm

    w = textnorm.fold_nasals(textnorm.fold_nukta(word))
    for suffix in ("ों", "्स"):
        if w.endswith(suffix) and len(w) > len(suffix) + 1:
            w = w[: -len(suffix)]
    return w


_LOANWORD_KEYS = None


def guess_language(text: str) -> tuple[str, str]:
    """Suggest hi / hi-en / sa from a reference transcript, with the evidence used.

    A suggestion for the user to confirm, not language identification:
      sa    - long words with Sanskrit endings and almost no Hindi function words;
      hi-en - Latin-script words, or English words written in Devanagari (3+ and 10%+ of words);
      hi    - otherwise.
    """
    global _LOANWORD_KEYS
    import textnorm

    words = (text or "").split()
    if not words:
        return "hi", "no reference text"
    latin, latin_share = textnorm.latin_share(text)
    deva = _deva_words(text)
    if deva:
        function_rate = sum(w in _HINDI_FUNCTION_WORDS for w in deva) / len(deva)
        avg_len = sum(len(w) for w in deva) / len(deva)
        ending_rate = sum(bool(_SANSKRIT_ENDINGS.search(w)) for w in deva) / len(deva)
        notebook_score = textnorm.sanskrit_likelihood(text)["sanskrit_score"]
        if len(deva) >= 5 and function_rate < 0.03 and (avg_len >= 6.0 or ending_rate >= 0.4 or notebook_score >= 0.5):
            return "sa", (f"Sanskrit cues: {function_rate:.0%} Hindi function words, average word "
                          f"{avg_len:.1f} letters, {ending_rate:.0%} Sanskrit-style endings")
    if _LOANWORD_KEYS is None:
        _LOANWORD_KEYS = {_loanword_key(w) for w in ENGLISH_IN_DEVANAGARI}
    english_deva = [w for w in deva if _loanword_key(w) in _LOANWORD_KEYS]
    english = latin + len(english_deva)
    share = english / len(words)
    unique = list(dict.fromkeys(english_deva))
    shown = ", ".join(unique[:8]) + (", …" if len(unique) > 8 else "")
    if english >= 3 and share >= 0.10:
        detail = f"{english} of {len(words)} words are English"
        detail += f" (Devanagari: {shown})" if english_deva else " (Latin script)"
        return "hi-en", detail
    return "hi", (f"Hindi; {english} English word(s) found, below the code-mix threshold" if english
                  else "Hindi; no Sanskrit or code-mix cues")


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


def media_data_uri(data: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def dependency_message(names=("HF_TOKEN", "SARVAM_API_KEY")) -> str | None:
    """Return a message for the first missing secret among `names` (None when all are set)."""
    for name in names:
        if not secret(name):
            return f"{name} is missing. Add it to Streamlit secrets or the deployment environment."
    return None


def missing_secrets(names) -> list[str]:
    return [name for name in names if not secret(name)]


@st.cache_resource(show_spinner="Loading Koyal Hindi ASR model…")
def load_koyal(device: str = "cpu"):
    import nemo.collections.asr as nemo_asr

    checkpoint = hf_hub_download(
        KOYAL_MODEL_ID, KOYAL_CHECKPOINT, revision=KOYAL_REVISION, token=secret("HF_TOKEN")
    )
    return nemo_asr.models.ASRModel.restore_from(checkpoint, map_location=device).eval()


def decode_audio(raw: bytes, suffix: str, max_seconds: float, return_info: bool = False):
    """Decode to 16 kHz mono float32 and scale the peak down to 0.95 if louder.

    Returns (audio, sample_rate, duration); with return_info=True also a dict holding
    measurements taken BEFORE peak scaling (clipping fraction, original peak), because
    scaling would otherwise hide clipped samples.
    """
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
    info = {
        "clipping_fraction": float(np.mean(np.abs(audio) >= CLIP_LEVEL)),
        "input_peak": peak,
        "peak_scaled": peak > 0.95,
    }
    if peak:
        audio = np.clip(audio * min(1.0, 0.95 / peak), -1.0, 1.0)
    if return_info:
        return audio, sample_rate, duration, info
    return audio, sample_rate, duration


@st.cache_data(show_spinner=False, max_entries=2)
def audio_quality_preview(raw: bytes, suffix: str, max_seconds: float) -> dict[str, float]:
    audio, _, duration, info = decode_audio(raw, suffix, max_seconds, return_info=True)
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0
    return {
        "duration": duration,
        "peak": float(info["input_peak"]),
        "rms_dbfs": float(20 * np.log10(max(rms, 1e-8))),
        "clipping_fraction": float(info["clipping_fraction"]),
    }


# ---------------------------------------------------------------- VAD and ASR windows
# Moved here from research.py so Live mode can use the same speech-aware windows.


@st.cache_resource(show_spinner=False)
def load_vad():
    from silero_vad import load_silero_vad

    return load_silero_vad()


def run_vad(y: np.ndarray, sr: int) -> list[dict]:
    from silero_vad import get_speech_timestamps

    stamps = get_speech_timestamps(
        torch.from_numpy(np.asarray(y, np.float32)),
        load_vad(),
        sampling_rate=sr,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=150,
        speech_pad_ms=120,
    )
    return [{"start": int(s["start"]), "end": int(s["end"])} for s in stamps]


def build_windows(n: int, sr: int, segments: list[dict], max_seconds: float = LIVE_ASR_WINDOW_SECONDS) -> list[tuple[int, int]]:
    """Windows of at most `max_seconds`, cut at the last VAD speech end at least 2 s in."""
    max_len = int(max_seconds * sr)
    vad_ends = sorted({min(n, s["end"]) for s in segments if 0 < s["end"] <= n})
    windows, start = [], 0
    while start < n:
        hard_end = min(n, start + max_len)
        if hard_end >= n:
            end = n
        else:
            legal = [b for b in vad_ends if start + 2 * sr <= b <= hard_end]
            end = max(legal) if legal else hard_end
        windows.append((start, end))
        start = end
    return windows


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


def speech_windows(audio: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    """VAD-aware windows; falls back to fixed 28 s cuts if VAD is unavailable."""
    if len(audio) <= LIVE_ASR_WINDOW_SECONDS * sample_rate:
        return [(0, len(audio))]
    try:
        segments = run_vad(audio, sample_rate)
    except Exception:
        segments = []
    return build_windows(len(audio), sample_rate, segments)


def transcribe(audio: np.ndarray, sample_rate: int, device: str = "cpu") -> tuple[str, float]:
    """Koyal transcript and inference seconds (model download/loading excluded)."""
    model = load_koyal(device)
    windows = speech_windows(audio, sample_rate)
    segments = [np.ascontiguousarray(audio[start:end], dtype=np.float32) for start, end in windows]
    started = time.perf_counter()
    text = " ".join(nemo_transcribe_segments(model, segments, sample_rate)).strip()
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


def score_transcript(reference_text: str, transcript: str, language: str = "hi") -> dict:
    """WER/CER under the notebook's scoring protocol (textnorm.score_all), in percent.

    wer / cer          normalized: numbers to words, nukta and nasal folding, acronyms, loanwords
    wer_raw / cer_raw  NFC + lowercase + punctuation removed only
    wer_script_tolerant  normalized, with cross-script words matched by phonetic key
    primary            the headline metric for the language (hi: wer, hi-en: script-tolerant, sa: cer)
    Error and denominator counts are returned so batches can pool corpus WER/CER with CIs.
    """
    import textnorm

    norm_language = "hi" if language in ("hi", "hi-en") else language
    reference_text = ZERO_WIDTH.sub("", reference_text or "")
    transcript = ZERO_WIDTH.sub("", transcript or "")
    if not normalize_reference_text(reference_text):
        raise ValueError("The reference transcript is empty after normalization.")
    scores = textnorm.score_all(reference_text, transcript, norm_language)
    reference = textnorm.normalize_text(reference_text, norm_language)
    hypothesis = textnorm.normalize_text(transcript, norm_language)
    if not reference:
        raise ValueError("The reference transcript is empty after normalization.")
    ops = align_words(reference.split(), hypothesis.split())
    result = {
        "wer": scores["wer_norm"] * 100,
        "cer": scores["cer_norm"] * 100,
        "wer_raw": scores["wer_raw"] * 100,
        "cer_raw": scores["cer_raw"] * 100,
        "wer_script_tolerant": scores["wer_script_tolerant"] * 100,
        "cross_script_matches": scores["cross_script_matches"],
        "substitutions": sum(op == "sub" for op, _, _ in ops),
        "deletions": sum(op == "del" for op, _, _ in ops),
        "insertions": sum(op == "ins" for op, _, _ in ops),
        "reference_words": len(reference.split()),
        "errors_norm": scores["errors_norm"],
        "errors_script_tolerant": scores["errors_script_tolerant"],
        "reference_words_norm": scores["reference_words_norm"],
        "char_errors_norm": scores["char_errors_norm"],
        "reference_chars_norm": scores["reference_chars_norm"],
        "language": language,
        "primary_metric": PRIMARY_METRIC.get(language, "wer"),
        "alignment": ops,
    }
    result["primary"] = result[result["primary_metric"]]
    return result
