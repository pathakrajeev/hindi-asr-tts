import ctypes
import gc
import html
import io
import logging
import re
import sys
import tempfile
import time
from pathlib import Path

import altair as alt
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import requests
import soundfile as sf
import streamlit as st
import torch

from speech_core import (
    SAMPLE_RATE,
    decode_audio,
    load_koyal,
    nemo_transcribe_segments,
    normalize_reference_text,
    score_transcript,
    secret,
    synthesize,
)

logger = logging.getLogger("hindi_asr_tts")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.propagate = False

MAX_RESEARCH_SECONDS = 120
MAX_WINDOW_SECONDS = 28.0
TTS_COMPARISON_MAX_CHARS = 650
SRAVAANI_MIN_INFERENCE_HEADROOM_MB = 1024
TARGET_RMS_DBFS = -20.0
PEAK_CEILING_DBFS = -1.0
DENOISE_SNR_THRESHOLD_DB = 20.0
DENOISE_STRENGTH = 0.60
DENOISE_MIN_IMPROVEMENT_DB = 1.0
MIN_NOISE_SECONDS = 0.50
PITCH_FLOOR_HZ, PITCH_CEILING_HZ = 60.0, 500.0
PROSODY_HOP = 160
MIN_PAUSE_S = 0.15
PITCH_TRANSITION_ST = 0.25
CONTOUR_POINTS = 200
TTS_SEED = 20260906
WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
WHISPER_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
INDIC_CONFORMER_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
INDIC_CONFORMER_REVISION = "2a77d305038dfc502fd9464e7d028b90381976cb"
SRAVAANI_MODEL_ID = "ARTPARK-IISc/SraVaani-1.0"
SRAVAANI_MODEL_REVISION = "f5dd5358325a5208775b91dad98918e079ea2b27"
MMS_MODEL_ID = "facebook/mms-tts-hin"
MMS_REVISION = "1d83b223ec78e30b944f7d96bd117eb3d7023303"
PARLER_MODEL_ID = "ai4bharat/indic-parler-tts"
PARLER_REVISION = "660bc53c64ae83150613264111708a0465211237"
PARLER_DESCRIPTION_MODEL_REVISIONS = {
    "google/flan-t5-large": "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
}
PARLER_DESCRIPTION = (
    "Divya, a female speaker, speaks Hindi at a moderate pace with balanced pitch and slight "
    "expressiveness. The recording is very clear, close-sounding, and has almost no background noise."
)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REFERENCE_COLOR = "#52514e"

ASR_ENGINES = {
    "koyal": {"label": "Koyal Hindi 120M", "memory": "~0.5 GB", "heavy": False},
    "sarvam-saaras-v3": {"label": "Sarvam Saaras v3", "memory": "API", "heavy": False},
    "whisper-large-v3-turbo": {"label": "Whisper Large-v3-Turbo", "memory": "~3.2 GB", "heavy": True},
    "indic-conformer-ctc": {"label": "IndicConformer 600M (CTC)", "memory": "~2.5 GB", "heavy": True},
    "indic-conformer-rnnt": {"label": "IndicConformer 600M (RNNT)", "memory": "shares CTC model", "heavy": True},
    "sravaani": {"label": "SraVaani-1.0", "memory": "~2–3 GB", "heavy": True},
}
TTS_ENGINES = {
    "bulbul-v3": {"label": "Sarvam Bulbul v3", "memory": "API", "heavy": False},
    "mms-tts-hin": {"label": "Meta MMS-TTS Hindi", "memory": "~0.15 GB", "heavy": False},
    "indic-parler-tts": {"label": "Indic Parler-TTS", "memory": "~3.5 GB", "heavy": True},
}
ASR_COLORS = {key: PALETTE[index] for index, key in enumerate(ASR_ENGINES)}
INPUT_LABEL = "Input speech"
TTS_COLORS = {INPUT_LABEL: REFERENCE_COLOR} | {
    TTS_ENGINES[key]["label"]: PALETTE[index] for index, key in enumerate(TTS_ENGINES)
}


# ---------------------------------------------------------------- model loaders


@st.cache_resource(show_spinner=False)
def load_vad():
    from silero_vad import load_silero_vad

    return load_silero_vad()


@st.cache_resource(show_spinner=False)
def load_whisper():
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID,
        revision=WHISPER_REVISION,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).eval()
    return model, AutoProcessor.from_pretrained(WHISPER_MODEL_ID, revision=WHISPER_REVISION)


@st.cache_resource(show_spinner=False)
def load_indic_conformer():
    from transformers import AutoModel

    return AutoModel.from_pretrained(
        INDIC_CONFORMER_MODEL_ID,
        revision=INDIC_CONFORMER_REVISION,
        trust_remote_code=True,
        token=secret("HF_TOKEN") or None,
    )


def _process_rss_mb() -> str:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return f"{int(line.split()[1]) / 1024:.0f} MB"
    except OSError:
        pass
    return "unavailable"


def _container_memory_mb() -> str:
    memory_files = (
        (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max")),
        (Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")),
    )
    for usage_path, limit_path in memory_files:
        try:
            usage = int(usage_path.read_text().strip())
            limit_text = limit_path.read_text().strip()
            if limit_text == "max":
                return f"{usage / 1024**2:.0f} MB / unlimited"
            limit = int(limit_text)
            if limit > 0 and limit < 1 << 60:
                return f"{usage / 1024**2:.0f} MB / {limit / 1024**2:.0f} MB"
        except (OSError, ValueError):
            continue
    return "unavailable"


def _container_memory_available_mb() -> int | None:
    memory_files = (
        (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max")),
        (Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")),
    )
    for usage_path, limit_path in memory_files:
        try:
            usage = int(usage_path.read_text().strip())
            limit_text = limit_path.read_text().strip()
            if limit_text == "max":
                return None
            limit = int(limit_text)
            if 0 < limit < 1 << 60:
                return max(0, (limit - usage) // 1024**2)
        except (OSError, ValueError):
            continue
    return None


def _trim_process_heap() -> str:
    if not sys.platform.startswith("linux"):
        return "unsupported"
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return "unavailable"
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return "trimmed free heap pages" if trim(0) else "no free heap pages"


@st.cache_resource(show_spinner=False)
def load_sravaani():
    from transformers import AutoModel

    logger.info(
        "SraVaani model load started model=%s revision=%s rss=%s container_memory=%s",
        SRAVAANI_MODEL_ID,
        SRAVAANI_MODEL_REVISION,
        _process_rss_mb(),
        _container_memory_mb(),
    )
    try:
        model = AutoModel.from_pretrained(
            SRAVAANI_MODEL_ID,
            revision=SRAVAANI_MODEL_REVISION,
            trust_remote_code=True,
            token=secret("HF_TOKEN") or None,
        ).eval()
    except Exception:
        logger.exception("SraVaani model load failed")
        raise
    for module in model.modules():
        if isinstance(module, torch.nn.RNNBase):
            module.flatten_parameters()
    logger.info(
        "SraVaani model load completed rss=%s container_memory=%s",
        _process_rss_mb(),
        _container_memory_mb(),
    )
    return model


@st.cache_resource(show_spinner=False)
def load_mms():
    from transformers import VitsModel, VitsTokenizer

    model = VitsModel.from_pretrained(
        MMS_MODEL_ID, revision=MMS_REVISION, dtype=torch.float32
    ).eval()
    tokenizer = VitsTokenizer.from_pretrained(MMS_MODEL_ID, revision=MMS_REVISION)
    return model, tokenizer


@st.cache_resource(show_spinner=False)
def load_parler():
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Indic Parler-TTS needs the parler-tts package, which pins transformers 4.46 and a "
            "torchaudio build that this environment (torch 2.14, transformers 5.x) cannot use. "
            "Run it from a separate Python 3.11 environment."
        ) from exc
    from transformers import AutoTokenizer

    token = secret("HF_TOKEN") or None
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        PARLER_MODEL_ID, revision=PARLER_REVISION, token=token
    ).eval()
    prompt_tokenizer = AutoTokenizer.from_pretrained(
        PARLER_MODEL_ID, revision=PARLER_REVISION, token=token
    )
    description_model_id = model.config.text_encoder._name_or_path
    description_revision = PARLER_DESCRIPTION_MODEL_REVISIONS.get(description_model_id)
    if description_revision is None:
        raise RuntimeError(
            f"No pinned tokenizer revision is configured for Indic Parler-TTS text encoder "
            f"{description_model_id!r}."
        )
    description_tokenizer = AutoTokenizer.from_pretrained(
        description_model_id, revision=description_revision, token=token
    )
    return model, prompt_tokenizer, description_tokenizer


@st.cache_resource(show_spinner=False)
def load_utmos():
    return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()


RELEASERS = {
    "koyal": load_koyal,
    "whisper-large-v3-turbo": load_whisper,
    "indic-conformer-ctc": load_indic_conformer,
    "indic-conformer-rnnt": load_indic_conformer,
    "sravaani": load_sravaani,
    "indic-parler-tts": load_parler,
}


def release(key: str) -> None:
    loader = RELEASERS.get(key)
    if loader is not None:
        loader.clear()
        collected = gc.collect()
        heap = _trim_process_heap()
        logger.info(
            "Released model cache key=%s gc_objects=%d heap_trim=%s rss=%s container_memory=%s",
            key,
            collected,
            heap,
            _process_rss_mb(),
            _container_memory_mb(),
        )


# ---------------------------------------------------------------- preprocessing


def rms_dbfs(y: np.ndarray) -> float:
    return float(20 * np.log10(max(float(np.sqrt(np.mean(np.square(y), dtype=np.float64) + 1e-12)), 1e-12)))


def normalize_rms(y: np.ndarray) -> np.ndarray:
    z = np.asarray(y, np.float32) * 10 ** ((TARGET_RMS_DBFS - rms_dbfs(y)) / 20)
    ceiling = 10 ** (PEAK_CEILING_DBFS / 20)
    peak = float(np.max(np.abs(z))) if len(z) else 0.0
    if peak > ceiling:
        z = z / peak * ceiling
    return z.astype(np.float32)


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


def speech_mask(n: int, segments: list[dict]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for segment in segments:
        mask[max(0, segment["start"]):min(n, segment["end"])] = True
    return mask


def estimate_snr_db(y: np.ndarray, mask: np.ndarray, sr: int) -> float:
    frame = max(1, int(0.025 * sr))
    speech_power, noise_power = [], []
    for start in range(0, len(y) - frame + 1, frame):
        part = np.asarray(y[start:start + frame], np.float64)
        fraction = float(mask[start:start + frame].mean())
        power = float(np.mean(part * part))
        if fraction >= 0.9:
            speech_power.append(power)
        elif fraction <= 0.1 and power > 1e-10:
            noise_power.append(power)
    if len(noise_power) * frame / sr < MIN_NOISE_SECONDS or not speech_power:
        return float("nan")
    noise = float(np.median(noise_power))
    return float(10 * np.log10(max(float(np.median(speech_power)) - noise, 1e-12) / max(noise, 1e-12)))


def adaptive_denoise(y: np.ndarray, sr: int, segments: list[dict]) -> dict:
    mask = speech_mask(len(y), segments)
    snr_before = estimate_snr_db(y, mask, sr)
    noise = y[~mask]
    result = {"snr_before": snr_before, "snr_after": snr_before, "decision": "skipped", "audio": y}
    if not np.isfinite(snr_before):
        result["reason"] = "SNR not estimable from the speech/non-speech regions."
    elif snr_before >= DENOISE_SNR_THRESHOLD_DB:
        result["reason"] = f"Input SNR {snr_before:.1f} dB is at or above the {DENOISE_SNR_THRESHOLD_DB:.0f} dB threshold."
    elif len(noise) / sr < MIN_NOISE_SECONDS:
        result["reason"] = f"Only {len(noise) / sr:.2f}s of non-speech audio to estimate noise ({MIN_NOISE_SECONDS}s needed)."
    else:
        import noisereduce as nr

        candidate = nr.reduce_noise(y=y, sr=sr, y_noise=noise, stationary=False, prop_decrease=DENOISE_STRENGTH)
        candidate = candidate.astype(np.float32)
        snr_candidate = estimate_snr_db(candidate, mask, sr)
        gain = snr_candidate - snr_before if np.isfinite(snr_candidate) else float("nan")
        if np.isfinite(gain) and gain >= DENOISE_MIN_IMPROVEMENT_DB:
            result.update(decision="applied", audio=candidate, snr_after=snr_candidate,
                          reason=f"SNR improved {snr_before:.1f} → {snr_candidate:.1f} dB.")
        else:
            result.update(decision="reverted", reason=(
                f"Denoised candidate improved SNR by {gain:+.1f} dB, below the +{DENOISE_MIN_IMPROVEMENT_DB:.0f} dB "
                "safety gate; original kept."
            ))
    return result


def build_windows(n: int, sr: int, segments: list[dict]) -> list[tuple[int, int]]:
    max_len = int(MAX_WINDOW_SECONDS * sr)
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


def preprocess(y: np.ndarray, sr: int, denoise: bool) -> dict:
    normalized = normalize_rms(y)
    segments = run_vad(normalized, sr)
    cleaned = adaptive_denoise(normalized, sr, segments) if denoise else {
        "snr_before": estimate_snr_db(normalized, speech_mask(len(normalized), segments), sr),
        "decision": "disabled", "reason": "Denoising turned off.", "audio": normalized,
    }
    cleaned.setdefault("snr_after", cleaned["snr_before"])
    mask = speech_mask(len(normalized), segments)
    return {
        "normalized": normalized,
        "processed": cleaned.pop("audio"),
        "vad": segments,
        "windows": build_windows(len(normalized), sr, segments),
        "speech_fraction": float(mask.mean()) if len(mask) else 0.0,
        "clipping_fraction": float(np.mean(np.abs(y) >= 0.999)),
        "input_rms_dbfs": rms_dbfs(y),
        **cleaned,
    }


# ---------------------------------------------------------------- ASR engines


def _segments(y: np.ndarray, windows: list[tuple[int, int]]) -> list[np.ndarray]:
    return [np.ascontiguousarray(y[start:end], dtype=np.float32) for start, end in windows]


def _text_from(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(_text_from(v) for v in value).strip()
    if isinstance(value, dict):
        for key in ("text", "transcript", "transcription"):
            if key in value:
                return _text_from(value[key])
    return str(getattr(value, "text", value) or "").strip()


def asr_koyal(y, sr, windows, options):
    return " ".join(nemo_transcribe_segments(load_koyal(), _segments(y, windows), sr))


def asr_sarvam(y, sr, windows, options):
    texts = []
    for segment in _segments(y, windows):
        wav = io.BytesIO()
        sf.write(wav, segment, sr, format="WAV", subtype="PCM_16")
        try:
            response = requests.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": secret("SARVAM_API_KEY")},
                files={"file": ("audio.wav", wav.getvalue(), "audio/wav")},
                data={"model": "saaras:v3", "mode": options["sarvam_mode"], "language_code": "hi-IN"},
                timeout=120,
            )
        except requests.RequestException as exc:
            raise RuntimeError("Network error calling Sarvam ASR.") from exc
        if response.status_code == 401:
            raise RuntimeError("Sarvam ASR authentication failed. Check SARVAM_API_KEY.")
        if response.status_code != 200:
            raise RuntimeError(f"Sarvam ASR error ({response.status_code}).")
        texts.append(str(response.json().get("transcript") or "").strip())
    return " ".join(texts)


def asr_whisper(y, sr, windows, options):
    model, processor = load_whisper()
    texts = []
    for segment in _segments(y, windows):
        features = processor(segment, sampling_rate=sr, return_tensors="pt").input_features
        with torch.inference_mode():
            tokens = model.generate(features, language="hi", task="transcribe", max_new_tokens=384)
        texts.append(processor.batch_decode(tokens, skip_special_tokens=True)[0].strip())
    return " ".join(texts)


def _asr_indic_conformer(decoder):
    def run(y, sr, windows, options):
        model = load_indic_conformer()
        texts = []
        with torch.inference_mode():
            for segment in _segments(y, windows):
                texts.append(_text_from(model(torch.from_numpy(segment).unsqueeze(0), "hi", decoder)))
        return " ".join(texts)

    return run


def asr_sravaani(y, sr, windows, options):
    model = load_sravaani()
    available_mb = _container_memory_available_mb()
    if available_mb is not None and available_mb < SRAVAANI_MIN_INFERENCE_HEADROOM_MB:
        message = (
            f"SraVaani loaded, but inference was not started: only {available_mb} MB of container "
            f"memory remained; at least {SRAVAANI_MIN_INFERENCE_HEADROOM_MB} MB is required. "
            "Use a larger-memory deployment or deselect SraVaani."
        )
        logger.error("%s rss=%s container_memory=%s", message, _process_rss_mb(), _container_memory_mb())
        raise RuntimeError(message)
    paths = []
    try:
        segments = _segments(y, windows)
        logger.info(
            "SraVaani inference started segments=%d rss=%s container_memory=%s",
            len(segments),
            _process_rss_mb(),
            _container_memory_mb(),
        )
        for segment in segments:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
                paths.append(file.name)
            sf.write(paths[-1], segment, sr, subtype="PCM_16")
        with torch.inference_mode():
            hypotheses = model.transcribe(paths, return_hypotheses=True)
        if isinstance(hypotheses, tuple):
            hypotheses = hypotheses[0]
        transcript = " ".join(_text_from(h) for h in hypotheses)
        logger.info(
            "SraVaani inference completed rss=%s container_memory=%s",
            _process_rss_mb(),
            _container_memory_mb(),
        )
        return transcript
    except Exception:
        logger.exception("SraVaani inference failed")
        raise
    finally:
        for path in paths:
            Path(path).unlink(missing_ok=True)


ASR_RUNNERS = {
    "koyal": asr_koyal,
    "sarvam-saaras-v3": asr_sarvam,
    "whisper-large-v3-turbo": asr_whisper,
    "indic-conformer-ctc": _asr_indic_conformer("ctc"),
    "indic-conformer-rnnt": _asr_indic_conformer("rnnt"),
    "sravaani": asr_sravaani,
}


# ---------------------------------------------------------------- TTS engines


def sentence_chunks(text: str, max_chars: int) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[।॥!?.])\s+", text) if s.strip()]
    chunks, current = [], ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def sentence_safe_prefix(text: str, max_chars: int) -> tuple[str, bool]:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text, False
    prefix = text[:max_chars]
    cut = max(prefix.rfind(mark) for mark in "।॥!?.")
    if cut < max_chars // 2:
        cut = prefix.rfind(" ")
    return prefix[: cut + 1 if cut > 0 else max_chars].strip(), True


def tts_bulbul(text):
    wav, elapsed, _ = synthesize(text)
    audio, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=False)
    return np.asarray(audio, np.float32), int(sr), elapsed


def tts_mms(text):
    model, tokenizer = load_mms()
    if getattr(tokenizer, "is_uroman", False):
        raise RuntimeError("This MMS checkpoint needs uroman romanization, which is not installed.")
    sr = int(model.config.sampling_rate)
    silence = np.zeros(int(0.18 * sr), dtype=np.float32)
    parts = []
    started = time.perf_counter()
    torch.manual_seed(TTS_SEED)
    with torch.inference_mode():
        for chunk in sentence_chunks(text, 220):
            inputs = tokenizer(chunk, return_tensors="pt")
            if inputs["input_ids"].shape[-1] == 0:
                continue
            parts += [model(**inputs).waveform[0].float().cpu().numpy(), silence]
    if not parts:
        raise RuntimeError("MMS-TTS produced no audio for this text.")
    return np.concatenate(parts[:-1]).astype(np.float32), sr, time.perf_counter() - started


def tts_parler(text):
    model, prompt_tokenizer, description_tokenizer = load_parler()
    description = description_tokenizer(PARLER_DESCRIPTION, return_tensors="pt")
    prompt = prompt_tokenizer(text, return_tensors="pt")
    torch.manual_seed(TTS_SEED)
    started = time.perf_counter()
    with torch.inference_mode():
        audio = model.generate(
            input_ids=description.input_ids,
            attention_mask=description.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
            max_new_tokens=2048,
        )
    elapsed = time.perf_counter() - started
    return audio.float().cpu().numpy().squeeze().astype(np.float32), int(model.config.sampling_rate), elapsed


TTS_RUNNERS = {"bulbul-v3": tts_bulbul, "mms-tts-hin": tts_mms, "indic-parler-tts": tts_parler}


def predicted_utmos(audio: np.ndarray, sr: int) -> float:
    model = load_utmos()
    a = librosa.resample(audio, orig_sr=sr, target_sr=16000) if sr != 16000 else audio
    a = np.ascontiguousarray(a, dtype=np.float32)
    window, scores, weights = 10 * 16000, [], []
    with torch.inference_mode():
        for start in range(0, len(a), window):
            chunk = a[start:start + window]
            samples = len(chunk)
            if samples < 16000:
                chunk = np.pad(chunk, (0, 16000 - samples))
            scores.append(float(model(torch.from_numpy(chunk).unsqueeze(0), 16000)))
            weights.append(samples)
    return float(np.average(scores, weights=weights))


# ---------------------------------------------------------------- prosody


def _pitch_track(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import parselmouth

        pitch = parselmouth.Sound(np.asarray(y, np.float64), sampling_frequency=sr).to_pitch_ac(
            time_step=PROSODY_HOP / sr, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ
        )
        hz = np.asarray(pitch.selected_array["frequency"], float)
        return np.asarray(pitch.xs()), np.where(hz > 0, hz, np.nan)
    except ImportError:
        f0 = librosa.yin(y, fmin=PITCH_FLOOR_HZ, fmax=PITCH_CEILING_HZ, sr=sr, frame_length=1024, hop_length=PROSODY_HOP)
        valid = (f0 > PITCH_FLOOR_HZ * 1.05) & (f0 < PITCH_CEILING_HZ * 0.95)
        return librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=PROSODY_HOP), np.where(valid, f0, np.nan)


def _voice_quality(y: np.ndarray, sr: int, segments: list[dict]) -> dict:
    out = {"jitter_pct": float("nan"), "shimmer_pct": float("nan"), "hnr_db": float("nan")}
    if not segments:
        return out
    longest = max(segments, key=lambda s: s["end"] - s["start"])
    clip = np.asarray(y[longest["start"]:longest["end"]], np.float64)
    if len(clip) < int(0.5 * sr):
        return out
    try:
        import parselmouth
        from parselmouth.praat import call

        sound = parselmouth.Sound(clip, sampling_frequency=sr)
        points = call(sound, "To PointProcess (periodic, cc)", PITCH_FLOOR_HZ, PITCH_CEILING_HZ)
        shortest, longest_period = 1 / PITCH_CEILING_HZ, 1 / PITCH_FLOOR_HZ
        out["jitter_pct"] = 100 * call(points, "Get jitter (local)", 0, 0, shortest, longest_period, 1.3)
        out["shimmer_pct"] = 100 * call([sound, points], "Get shimmer (local)", 0, 0, shortest, longest_period, 1.3, 1.6)
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, PITCH_FLOOR_HZ, 0.1, 1.0)
        out["hnr_db"] = call(harmonicity, "Get mean", 0, 0)
    except Exception:
        pass
    return {key: float(value) if np.isfinite(value) else float("nan") for key, value in out.items()}


def analyze_prosody(y: np.ndarray, sr: int, segments: list[dict], text: str) -> dict:
    duration = len(y) / sr
    mask = speech_mask(len(y), segments)
    speech_seconds = float(mask.sum()) / sr
    f0_times, f0 = _pitch_track(y, sr)
    frame_in_speech = mask[np.clip((f0_times * sr).astype(int), 0, len(mask) - 1)] if len(f0_times) else np.array([], bool)
    voiced = np.isfinite(f0)
    voiced_values = f0[voiced]
    metrics = {
        "Duration (s)": duration,
        "Speech duration (s)": speech_seconds,
        "Speech rate (words/s)": len(normalize_reference_text(text).split()) / speech_seconds if speech_seconds else float("nan"),
        "Voiced ratio in speech": float(voiced[frame_in_speech].mean()) if frame_in_speech.any() else float("nan"),
    }
    if len(voiced_values) >= 3:
        p05, p95 = np.percentile(voiced_values, [5, 95])
        median = float(np.median(voiced_values))
        semitones = 12 * np.log2(voiced_values / median)
        metrics |= {
            "F0 median (Hz)": median,
            "F0 p05 (Hz)": float(p05),
            "F0 p95 (Hz)": float(p95),
            "F0 range (semitones)": float(12 * np.log2(p95 / p05)),
            "F0 std (semitones)": float(np.std(semitones)),
            "F0 slope (st/s)": float(np.polyfit(f0_times[voiced], semitones, 1)[0]),
        }
        indices = np.flatnonzero(voiced)
        consecutive = np.diff(indices) == 1
        if consecutive.any():
            step = 12 * np.log2(f0[indices[1:][consecutive]] / f0[indices[:-1][consecutive]])
            metrics |= {
                "Pitch change rate (st/s)": float(np.mean(np.abs(step)) / (PROSODY_HOP / sr)),
                "Rising transitions (%)": float(np.mean(step > PITCH_TRANSITION_ST) * 100),
                "Falling transitions (%)": float(np.mean(step < -PITCH_TRANSITION_ST) * 100),
            }
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=PROSODY_HOP)[0]
    energy_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=PROSODY_HOP)
    energy_db = 20 * np.log10(np.maximum(rms, 1e-10))
    energy_in_speech = mask[np.clip((energy_times * sr).astype(int), 0, len(mask) - 1)]
    speech_energy = energy_db[energy_in_speech]
    if len(speech_energy):
        e05, e95 = np.percentile(speech_energy, [5, 95])
        metrics |= {"Energy mean (dBFS)": float(np.mean(speech_energy)), "Energy dynamic range (dB)": float(e95 - e05)}
    pauses = np.array([
        (b["start"] - a["end"]) / sr for a, b in zip(segments, segments[1:]) if (b["start"] - a["end"]) / sr >= MIN_PAUSE_S
    ])
    metrics |= {
        "Pause count": int(len(pauses)),
        "Pause mean (s)": float(pauses.mean()) if len(pauses) else 0.0,
        "Internal pause ratio": float(pauses.sum() / duration) if duration else float("nan"),
    }
    voice = _voice_quality(y, sr, segments)
    metrics |= {"Jitter (%)": voice["jitter_pct"], "Shimmer (%)": voice["shimmer_pct"], "HNR (dB)": voice["hnr_db"]}
    return {"metrics": metrics, "f0_t": f0_times, "f0": f0, "energy_t": energy_times, "energy_db": energy_db}


def _normalized_contour(times, values, semitone: bool) -> np.ndarray | None:
    times, values = np.asarray(times, float), np.asarray(values, float)
    valid = np.isfinite(values) & np.isfinite(times)
    if valid.sum() < 10 or np.ptp(times[valid]) <= 0:
        return None
    t, v = times[valid], values[valid]
    if semitone:
        v = 12 * np.log2(v / np.median(v))
    grid = np.interp(np.linspace(0, 1, CONTOUR_POINTS), (t - t.min()) / np.ptp(t), v)
    spread = np.std(grid)
    return (grid - grid.mean()) / spread if spread > 0 else None


def _corr(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a is not None and b is not None else float("nan")


def compare_prosody(reference: dict, candidate: dict) -> dict:
    rm, cm = reference["metrics"], candidate["metrics"]

    def ratio(key):
        r, c = rm.get(key, float("nan")), cm.get(key, float("nan"))
        return c / r if np.isfinite(r) and r > 0 and np.isfinite(c) else float("nan")

    def abs_diff(key):
        return abs(cm.get(key, float("nan")) - rm.get(key, float("nan")))

    return {
        "F0 shape correlation": _corr(
            _normalized_contour(reference["f0_t"], reference["f0"], True),
            _normalized_contour(candidate["f0_t"], candidate["f0"], True),
        ),
        "Energy contour correlation": _corr(
            _normalized_contour(reference["energy_t"], reference["energy_db"], False),
            _normalized_contour(candidate["energy_t"], candidate["energy_db"], False),
        ),
        "Pitch range ratio": ratio("F0 range (semitones)"),
        "Duration ratio": ratio("Duration (s)"),
        "Speech rate ratio": ratio("Speech rate (words/s)"),
        "Pause ratio |diff|": abs_diff("Internal pause ratio"),
        "Voiced ratio |diff|": abs_diff("Voiced ratio in speech"),
    }


# ---------------------------------------------------------------- pipeline


def _to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    return librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE) if sr != SAMPLE_RATE else audio


def _wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def run_research(audio, sr, reference_text, options, status) -> dict:
    log = []

    def step(name, fn, *args):
        status.write(f"{name}…")
        started = time.perf_counter()
        logger.info(
            "Research step started name=%s rss=%s container_memory=%s",
            name,
            _process_rss_mb(),
            _container_memory_mb(),
        )
        try:
            value = fn(*args)
            elapsed = time.perf_counter() - started
            log.append({"Step": name, "Status": "ok", "Seconds": elapsed, "Detail": ""})
            logger.info(
                "Research step completed name=%s seconds=%.2f rss=%s container_memory=%s",
                name,
                elapsed,
                _process_rss_mb(),
                _container_memory_mb(),
            )
            return value, None
        except Exception as exc:
            message = f"{type(exc).__name__}: {str(exc)[:400]}"
            elapsed = time.perf_counter() - started
            log.append({"Step": name, "Status": "failed", "Seconds": elapsed, "Detail": message})
            logger.exception(
                "Research step failed name=%s seconds=%.2f rss=%s container_memory=%s",
                name,
                elapsed,
                _process_rss_mb(),
                _container_memory_mb(),
            )
            return None, message

    pre, error = step("Preprocessing (normalize, VAD, denoise)", preprocess, audio, sr, options["denoise"])
    if error:
        raise RuntimeError(f"Preprocessing failed: {error}")
    duration = len(audio) / sr

    asr_rows = []
    for key in options["asr"]:
        started = time.perf_counter()
        text, error = step(f"ASR · {ASR_ENGINES[key]['label']}", ASR_RUNNERS[key], pre["processed"], sr, pre["windows"], options)
        elapsed = time.perf_counter() - started
        row = {"key": key, "label": ASR_ENGINES[key]["label"], "status": "failed" if error else "ok", "error": error,
               "text": text or "", "latency": elapsed, "rtf": elapsed / duration}
        if not error:
            if not row["text"]:
                row |= {"status": "failed", "error": "Model returned an empty transcript."}
            else:
                row |= score_transcript(reference_text, row["text"])
        asr_rows.append(row)
        if options["release"]:
            release(key)

    tts_text, source = reference_text, "Reference transcript"
    if options["tts_source"] != "reference":
        source_row = next((r for r in asr_rows if r["key"] == options["tts_source"] and r["status"] == "ok"), None)
        if source_row:
            tts_text, source = source_row["text"], f"{source_row['label']} transcript"
        else:
            log.append({"Step": "TTS text", "Status": "fallback", "Seconds": 0.0,
                        "Detail": "Selected ASR transcript unavailable; using the reference transcript."})
    tts_text, truncated = sentence_safe_prefix(tts_text, TTS_COMPARISON_MAX_CHARS)

    tts_rows = []
    for key in options["tts"]:
        result, error = step(f"TTS · {TTS_ENGINES[key]['label']}", TTS_RUNNERS[key], tts_text)
        row = {"key": key, "label": TTS_ENGINES[key]["label"], "status": "failed" if error else "ok", "error": error}
        if result:
            out, out_sr, elapsed = result
            out_duration = len(out) / out_sr
            row |= {"audio": out, "sr": out_sr, "wav": _wav_bytes(out, out_sr), "latency": elapsed,
                    "duration": out_duration, "rtf": elapsed / out_duration if out_duration else float("nan")}
            if not np.isfinite(out).all() or not len(out):
                row |= {"status": "failed", "error": "Model returned empty or invalid audio."}
        tts_rows.append(row)
        if options["release"]:
            release(key)

    judge = options["judge"]
    for row in (r for r in tts_rows if r["status"] == "ok"):
        speech = normalize_rms(_to_16k(row["audio"], row["sr"]))
        row["vad"], _ = step(f"VAD · {row['label']} output", run_vad, speech, SAMPLE_RATE)
        row["audio16"] = speech
        if judge:
            text, error = step(f"Loopback · {row['label']} → {ASR_ENGINES[judge]['label']}",
                               ASR_RUNNERS[judge], speech, SAMPLE_RATE,
                               build_windows(len(speech), SAMPLE_RATE, row["vad"] or []), options)
            row["loopback_text"] = text or ""
            row["loopback_error"] = error
            if text:
                row["loopback"] = score_transcript(tts_text, text)
        if options["utmos"]:
            row["utmos"], row["utmos_error"] = step(f"UTMOS · {row['label']}", predicted_utmos, row["audio"], row["sr"])
    if judge and options["release"]:
        release(judge)
    if options["utmos"]:
        load_utmos.clear()
        gc.collect()

    prosody = {}
    value, _ = step("Prosody · input speech", analyze_prosody, pre["normalized"], sr, pre["vad"], reference_text)
    if value:
        prosody[INPUT_LABEL] = value
    for row in (r for r in tts_rows if r["status"] == "ok"):
        value, _ = step(f"Prosody · {row['label']}", analyze_prosody, row["audio16"], SAMPLE_RATE, row.get("vad") or [], tts_text)
        if value:
            prosody[row["label"]] = value

    return {"audio": audio, "sr": sr, "reference": reference_text, "pre": pre, "asr": asr_rows,
            "tts_text": tts_text, "tts_source": source, "tts_truncated": truncated, "tts": tts_rows,
            "judge": ASR_ENGINES[judge]["label"] if judge else None, "prosody": prosody, "log": log}


# ---------------------------------------------------------------- charts


def _bar(frame: pd.DataFrame, field: str, title: str, colors: dict, color_field: str, fmt: str = ".2f", rule: float | None = None):
    frame = frame.dropna(subset=[field])
    if frame.empty:
        return None
    order = [k for k in colors if k in set(frame[color_field])]
    bars = alt.Chart(frame, title=title).mark_bar(cornerRadiusEnd=4, height={"band": 0.6}).encode(
        x=alt.X(f"{field}:Q", title=None),
        y=alt.Y(f"{color_field}:N", sort=order, title=None),
        color=alt.Color(f"{color_field}:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())), legend=None),
        tooltip=[alt.Tooltip(f"{color_field}:N", title="Model"), alt.Tooltip(f"{field}:Q", title=title, format=fmt)],
    )
    labels = bars.mark_text(align="left", dx=4, color="#52514e").encode(text=alt.Text(f"{field}:Q", format=fmt), color=alt.value("#52514e"))
    chart = bars + labels
    if rule is not None:
        chart += alt.Chart(pd.DataFrame({"x": [rule]})).mark_rule(strokeDash=[4, 4], color="#8a8985").encode(x="x:Q")
    return chart.properties(height=max(90, 34 * len(frame)))


def _show(chart):
    if chart is not None:
        st.altair_chart(chart, width="stretch")


def _fig_close(fig, alt_text):
    st.pyplot(fig, alt=alt_text)
    plt.close(fig)


def _mel_db(y, sr, ref):
    return librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=512, hop_length=160, fmax=sr / 2),
        ref=ref, top_db=80,
    )


def render_preprocessing(result):
    pre, sr = result["pre"], result["sr"]
    columns = st.columns(4)
    columns[0].metric("Duration", f"{len(result['audio']) / sr:.1f} s")
    columns[1].metric("Speech in clip", f"{pre['speech_fraction'] * 100:.0f}%")
    snr = pre["snr_before"]
    columns[2].metric("SNR before", f"{snr:.1f} dB" if np.isfinite(snr) else "n/a")
    after = pre["snr_after"]
    columns[3].metric("SNR after", f"{after:.1f} dB" if np.isfinite(after) else "n/a")
    st.markdown(f"**Denoising: {pre['decision']}** — {pre['reason']}")
    st.caption(
        f"Input RMS {pre['input_rms_dbfs']:.1f} dBFS → normalized to {TARGET_RMS_DBFS:.0f} dBFS · "
        f"clipped samples {pre['clipping_fraction'] * 100:.2f}% · {len(pre['vad'])} speech segments · "
        f"{len(pre['windows'])} ASR window(s) of ≤{MAX_WINDOW_SECONDS:.0f}s cut at speech boundaries."
    )

    y = pre["normalized"]
    t = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
    ax.plot(t, y, linewidth=0.6, color=PALETTE[0])
    for index, segment in enumerate(pre["vad"]):
        ax.axvspan(segment["start"] / sr, segment["end"] / sr, color=PALETTE[2], alpha=0.18,
                   label="Speech (SileroVAD)" if index == 0 else None)
    ax.set(title="Normalized waveform with detected speech", xlabel="Time (s)", ylabel="Amplitude")
    ax.grid(alpha=0.2)
    if pre["vad"]:
        ax.legend(loc="upper right")
    _fig_close(fig, "Waveform of the normalized input with shaded speech regions found by voice activity detection")

    ref = max(float(np.max(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=512, hop_length=160))), 1e-10)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    for axis, signal, title in ((axes[0], y, "Before denoising"), (axes[1], pre["processed"], f"Sent to ASR ({pre['decision']})")):
        image = librosa.display.specshow(_mel_db(signal, sr, ref), sr=sr, hop_length=160, x_axis="time",
                                         y_axis="mel", ax=axis, vmin=-80, vmax=0, cmap="viridis")
        axis.set(title=f"Log-Mel spectrogram — {title}")
    fig.colorbar(image, ax=axes, format="%+2.0f dB")
    _fig_close(fig, "Log-Mel spectrograms before and after denoising on a shared decibel scale")


def _alignment_html(ops) -> str:
    parts = []
    for op, ref, hyp in ops:
        ref, hyp = html.escape(ref), html.escape(hyp)
        if op == "ok":
            parts.append(hyp)
        elif op == "sub":
            parts.append(f'<span style="background:rgba(235,104,52,.18);border-bottom:2px solid {PALETTE[1]}" '
                         f'title="substitution">{hyp} <small>({ref})</small></span>')
        elif op == "del":
            parts.append(f'<del style="text-decoration-color:{PALETTE[7]}" title="deletion">{ref}</del>')
        else:
            parts.append(f'<ins style="text-decoration-color:{PALETTE[6]}" title="insertion">{hyp}</ins>')
    return '<div style="line-height:2">' + " ".join(parts) + "</div>"


def render_asr(result):
    rows = result["asr"]
    ok = [r for r in rows if r["status"] == "ok"]
    table = pd.DataFrame([{
        "Model": r["label"], "Status": r["status"],
        "WER (%)": r.get("wer"), "CER (%)": r.get("cer"),
        "Subs": r.get("substitutions"), "Dels": r.get("deletions"), "Ins": r.get("insertions"),
        "Latency (s)": r["latency"], "RTF": r["rtf"],
    } for r in rows])
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.2f") for c in ("WER (%)", "CER (%)", "Latency (s)", "RTF")
    })
    for r in rows:
        if r["status"] != "ok":
            st.error(f"**{r['label']}** failed — {r['error']}")
    if not ok:
        return

    frame = pd.DataFrame([{"Model": r["label"], "WER": r["wer"], "CER": r["cer"], "Latency": r["latency"], "RTF": r["rtf"]} for r in ok])
    colors = {ASR_ENGINES[k]["label"]: c for k, c in ASR_COLORS.items()}
    left, right = st.columns(2)
    with left:
        _show(_bar(frame, "WER", "Word error rate (%) — lower is better", colors, "Model"))
    with right:
        _show(_bar(frame, "CER", "Character error rate (%) — lower is better", colors, "Model"))

    errors = pd.DataFrame([
        {"Model": r["label"], "Error type": kind, "Count": r[field]}
        for r in ok for kind, field in (("Substitution", "substitutions"), ("Deletion", "deletions"), ("Insertion", "insertions"))
    ])
    kinds = ["Substitution", "Deletion", "Insertion"]
    stacked = alt.Chart(errors, title="Word errors by type").mark_bar(stroke="white", strokeWidth=2, height={"band": 0.6}).encode(
        x=alt.X("Count:Q", stack="zero", title="Word errors"),
        y=alt.Y("Model:N", sort=[l for l in colors if l in set(errors["Model"])], title=None),
        color=alt.Color("Error type:N", scale=alt.Scale(domain=kinds, range=PALETTE[:3]), legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("order:Q"),
        tooltip=["Model", "Error type", "Count"],
    ).transform_calculate(order="indexof(['Substitution','Deletion','Insertion'], datum['Error type'])")
    _show(stacked.properties(height=max(90, 34 * len(ok))))

    left, right = st.columns(2)
    with left:
        _show(_bar(frame, "Latency", "Processing time (s)", colors, "Model"))
    with right:
        _show(_bar(frame, "RTF", "Real-time factor — below 1 is faster than real time", colors, "Model", fmt=".3f", rule=1.0))

    st.markdown("#### Word-level alignment against the reference")
    st.caption("Highlighted = substitution (reference in brackets) · strike-through = deleted reference word · underline = inserted word.")
    for r in ok:
        with st.expander(f"{r['label']} — WER {r['wer']:.1f}%"):
            st.markdown(_alignment_html(r["alignment"]), unsafe_allow_html=True)
            st.caption("Raw transcript")
            st.text(r["text"])


def render_tts(result):
    st.markdown(f"**Text sent to every TTS model** ({result['tts_source']}"
                f"{', trimmed at a sentence boundary to ' + str(TTS_COMPARISON_MAX_CHARS) + ' characters' if result['tts_truncated'] else ''}):")
    st.text(result["tts_text"])
    rows = result["tts"]
    st.audio(_wav_bytes(result["audio"], result["sr"]), format="audio/wav")
    st.caption("Original input speech")
    for r in rows:
        if r["status"] == "ok":
            st.markdown(f"**{r['label']}**")
            st.audio(r["wav"], format="audio/wav")
        else:
            st.error(f"**{r['label']}** failed — {r['error']}")
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return
    judge = result["judge"]
    table = pd.DataFrame([{
        "Model": r["label"], "Duration (s)": r["duration"], "Latency (s)": r["latency"], "RTF": r["rtf"],
        "Loopback WER (%)": r.get("loopback", {}).get("wer"), "Loopback CER (%)": r.get("loopback", {}).get("cer"),
        "Predicted UTMOS": r.get("utmos"),
    } for r in ok])
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.2f") for c in table.columns if c != "Model"
    })
    st.caption(
        (f"Loopback = the generated speech re-transcribed by **{judge}** and scored against the TTS input text "
         "(an intelligibility proxy). " if judge else "")
        + "UTMOS is a predicted naturalness score on a 1–5 scale, not a human MOS."
    )
    for r in ok:
        if r.get("loopback_error"):
            st.warning(f"Loopback for {r['label']} failed — {r['loopback_error']}")
        if r.get("utmos_error"):
            st.warning(f"UTMOS for {r['label']} failed — {r['utmos_error']}")

    colors = {k: v for k, v in TTS_COLORS.items() if k != INPUT_LABEL}
    frame = pd.DataFrame([{
        "Model": r["label"], "Latency": r["latency"], "RTF": r["rtf"], "Duration": r["duration"],
        "Loopback WER": r.get("loopback", {}).get("wer", np.nan), "UTMOS": r.get("utmos") if r.get("utmos") is not None else np.nan,
    } for r in ok])
    left, right = st.columns(2)
    with left:
        _show(_bar(frame, "Loopback WER", "Loopback WER (%) — lower is more intelligible", colors, "Model"))
        _show(_bar(frame, "Latency", "Synthesis time (s)", colors, "Model"))
    with right:
        _show(_bar(frame, "UTMOS", "Predicted UTMOS (1–5) — higher is better", colors, "Model"))
        _show(_bar(frame, "RTF", "Real-time factor — below 1 is faster than real time", colors, "Model", fmt=".3f", rule=1.0))

    signals = [(INPUT_LABEL, result["pre"]["normalized"], result["sr"])] + [(r["label"], r["audio16"], SAMPLE_RATE) for r in ok]
    fig, axes = plt.subplots(len(signals), 2, figsize=(13, 2.6 * len(signals)), constrained_layout=True, squeeze=False)
    for (label, signal, sr), (wave_axis, mel_axis) in zip(signals, axes):
        t = np.arange(len(signal)) / sr
        wave_axis.plot(t, signal, linewidth=0.6, color=TTS_COLORS[label])
        wave_axis.set(title=f"{label} — waveform", xlabel="Time (s)", ylabel="Amplitude")
        wave_axis.grid(alpha=0.2)
        ref = max(float(np.max(librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, n_fft=512, hop_length=160))), 1e-10)
        librosa.display.specshow(_mel_db(signal, sr, ref), sr=sr, hop_length=160, x_axis="time", y_axis="mel",
                                 ax=mel_axis, vmin=-80, vmax=0, cmap="viridis")
        mel_axis.set(title=f"{label} — log-Mel spectrogram")
    _fig_close(fig, "Waveform and log-Mel spectrogram for the input speech and each TTS output")


def render_prosody(result):
    prosody = result["prosody"]
    if not prosody:
        st.info("Prosody analysis did not produce results; see the run log.")
        return
    metrics = pd.DataFrame({label: analysis["metrics"] for label, analysis in prosody.items()})
    metrics.index.name = "Measure"
    st.dataframe(metrics, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in metrics.columns})
    st.caption(
        "Pitch via Praat autocorrelation (60–500 Hz). Jitter, shimmer and HNR come from the longest speech segment and are "
        "exploratory, not clinical. Input prosody uses the normalized recording before denoising."
    )

    if INPUT_LABEL in prosody and len(prosody) > 1:
        comparison = pd.DataFrame({
            label: compare_prosody(prosody[INPUT_LABEL], analysis) for label, analysis in prosody.items() if label != INPUT_LABEL
        })
        comparison.index.name = "Compared with input speech"
        st.markdown("#### How closely each TTS voice follows the input speaker")
        st.dataframe(comparison, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in comparison.columns})
        st.caption("Correlations compare normalized contour shapes over relative time (1 = same shape). Ratios near 1 and |diff| near 0 mean closer to the input.")

    order = [label for label in TTS_COLORS if label in prosody]
    scale = alt.Scale(domain=order, range=[TTS_COLORS[l] for l in order])
    pitch_rows, energy_rows = [], []
    for label, analysis in prosody.items():
        t = np.asarray(analysis["f0_t"], float)
        span = t.max() if len(t) and t.max() > 0 else 1.0
        pitch_rows += [{"Source": label, "Relative time": float(x / span), "F0 (Hz)": float(v)}
                       for x, v in zip(t, analysis["f0"]) if np.isfinite(v)]
        et = np.asarray(analysis["energy_t"], float)
        espan = et.max() if len(et) and et.max() > 0 else 1.0
        stride = max(1, len(et) // 400)
        energy_rows += [{"Source": label, "Relative time": float(x / espan), "Energy (dBFS)": float(v)}
                        for x, v in list(zip(et, analysis["energy_db"]))[::stride]]
    pitch, energy = pd.DataFrame(pitch_rows), pd.DataFrame(energy_rows)
    legend = alt.Legend(orient="top", title=None)
    if not pitch.empty:
        _show(alt.Chart(pitch, title="Pitch contour (voiced frames)").mark_circle(size=10, opacity=0.6).encode(
            x=alt.X("Relative time:Q", scale=alt.Scale(domain=[0, 1]), title="Relative time (0 = start, 1 = end)"),
            y=alt.Y("F0 (Hz):Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Source:N", scale=scale, legend=legend),
            tooltip=["Source", alt.Tooltip("Relative time:Q", format=".2f"), alt.Tooltip("F0 (Hz):Q", format=".0f")],
        ).properties(height=280))
        _show(alt.Chart(pitch, title="Pitch distribution").mark_boxplot(size=24, extent="min-max").encode(
            x=alt.X("F0 (Hz):Q", scale=alt.Scale(zero=False)),
            y=alt.Y("Source:N", sort=order, title=None),
            color=alt.Color("Source:N", scale=scale, legend=None),
        ).properties(height=max(90, 40 * len(order))))
    if not energy.empty:
        _show(alt.Chart(energy, title="Energy contour").mark_line(strokeWidth=2).encode(
            x=alt.X("Relative time:Q", scale=alt.Scale(domain=[0, 1]), title="Relative time (0 = start, 1 = end)"),
            y=alt.Y("Energy (dBFS):Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Source:N", scale=scale, legend=legend),
            tooltip=["Source", alt.Tooltip("Relative time:Q", format=".2f"), alt.Tooltip("Energy (dBFS):Q", format=".1f")],
        ).properties(height=260))


# ---------------------------------------------------------------- page


def _format_engine(engines):
    return lambda key: f"{engines[key]['label']} ({engines[key]['memory']})"


def render_research_tab():
    st.subheader("Research mode")
    st.caption(
        "Compare selected ASR and TTS models on one recording, scored against a "
        "golden-reference transcript. Local models load only when selected."
    )

    reference_file = st.file_uploader("Golden-reference transcript (.txt, UTF-8)", type=["txt"], key="research_reference")
    reference_text = ""
    if reference_file is not None:
        try:
            reference_text = reference_file.getvalue().decode("utf-8-sig").strip()
            with st.expander("Reference transcript"):
                st.text(reference_text)
        except UnicodeDecodeError:
            st.error("The reference file must be UTF-8 text.")
    else:
        reference_text = st.text_area("…or paste the reference transcript", height=90, key="research_reference_text").strip()
    reference_ok = bool(normalize_reference_text(reference_text))

    left, right = st.columns(2)
    with left:
        recording = st.audio_input("Record Hindi speech", key="research_recording")
    with right:
        upload = st.file_uploader("Or upload audio", type=["wav", "mp3", "m4a", "mp4", "flac", "ogg", "aac"], key="research_upload")
    source = recording or upload

    with st.container(border=True):
        st.markdown("**Models**")
        asr = st.multiselect("ASR models", list(ASR_ENGINES), default=["koyal", "sarvam-saaras-v3"],
                             format_func=_format_engine(ASR_ENGINES), key="research_asr")
        tts = st.multiselect("TTS models", list(TTS_ENGINES), default=["bulbul-v3", "mms-tts-hin"],
                             format_func=_format_engine(TTS_ENGINES), key="research_tts")
        columns = st.columns(2)
        tts_source = columns[0].selectbox(
            "Text for TTS", ["reference", *asr],
            format_func=lambda k: "Reference transcript" if k == "reference" else f"{ASR_ENGINES[k]['label']} transcript",
            key="research_tts_source",
        )
        judge = columns[1].selectbox(
            "Loopback judge (re-transcribes TTS output)", [None, *asr],
            index=1 if asr else 0,
            format_func=lambda k: "None" if k is None else ASR_ENGINES[k]["label"],
            key="research_judge",
        )
        columns = st.columns(4)
        sarvam_mode = columns[0].selectbox("Saaras mode", ["codemix", "transcribe"], key="research_sarvam_mode")
        denoise = columns[1].toggle("Adaptive denoise", value=True, key="research_denoise")
        utmos = columns[2].toggle("UTMOS score", value=True, key="research_utmos", help="Downloads the SpeechMOS UTMOS22 model on first use.")
        release_models = columns[3].toggle("Free models after use", value=True, key="research_release",
                                           help="Unloads local models after each ASR/TTS step to reduce peak memory. A selected loopback judge may be loaded again later.")

        heavy = [ASR_ENGINES[k]["label"] for k in asr if ASR_ENGINES[k]["heavy"]] + [TTS_ENGINES[k]["label"] for k in tts if TTS_ENGINES[k]["heavy"]]
        if heavy:
            st.warning(
                f"{', '.join(heavy)} run locally on CPU: expect several minutes per model and first-run downloads of several GB. "
                "If loading fails with a memory error, enable a Windows page file or deselect models. "
                "Streamlit Community Cloud (~2.7 GB RAM) cannot load these."
            )

    if source is None:
        st.info(f"Record or upload a Hindi utterance up to {MAX_RESEARCH_SECONDS} seconds.")
    elif not reference_ok:
        st.warning("Add the golden-reference transcript to enable scoring.")

    can_run = source is not None and reference_ok and bool(asr)
    if st.button("Run research comparison", type="primary", width="stretch", disabled=not can_run, key="research_run"):
        try:
            audio, sr, _ = decode_audio(source.getvalue(), Path(getattr(source, "name", "recording.wav")).suffix, MAX_RESEARCH_SECONDS)
            options = {"asr": asr, "tts": tts, "tts_source": tts_source, "judge": judge, "sarvam_mode": sarvam_mode,
                       "denoise": denoise, "utmos": utmos, "release": release_models}
            with st.status("Running research comparison…", expanded=True) as status:
                st.session_state["research_result"] = run_research(audio, sr, reference_text, options, status)
                st.session_state["research_result"]["name"] = getattr(source, "name", "recording")
                status.update(label="Research comparison complete", state="complete", expanded=False)
        except Exception as exc:
            st.error(f"Research run failed: {type(exc).__name__}: {exc}")

    result = st.session_state.get("research_result")
    if not result:
        return
    st.divider()
    st.markdown(f"### Results — {html.escape(str(result['name']))}")
    sections = st.tabs(["Audio preprocessing", "ASR comparison", "TTS comparison", "Prosody & voice", "Run log"])
    with sections[0]:
        render_preprocessing(result)
    with sections[1]:
        render_asr(result)
    with sections[2]:
        render_tts(result)
    with sections[3]:
        render_prosody(result)
    with sections[4]:
        st.dataframe(pd.DataFrame(result["log"]), hide_index=True, width="stretch",
                     column_config={"Seconds": st.column_config.NumberColumn(format="%.2f")})
