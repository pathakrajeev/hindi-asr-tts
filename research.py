import ctypes
import gc
import hashlib
import html
import io
import logging
import re
import shutil
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
    LANGUAGES,
    PRIMARY_LABEL,
    SAMPLE_RATE,
    audio_quality_preview,
    build_windows,
    decode_audio,
    guess_language,
    load_koyal,
    load_vad,
    media_data_uri,
    missing_secrets,
    nemo_transcribe_segments,
    normalize_reference_text,
    run_vad,
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
FEMALE_VOICE_F0_THRESHOLD_HZ = 165.0

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
    "maya2-native": {"label": "Maya 2 Native (local GPU)", "memory": "~9.2 GB VRAM peak", "heavy": True},
    "mms-tts-hin": {"label": "Meta MMS-TTS Hindi", "memory": "~0.15 GB", "heavy": False},
}
MAYA_FEMALE_VOICES = [
    "Ahana", "Vaidehi", "Malini", "Charvi", "Jahnavi",
    "Nivedita", "Ishwari", "Vrinda", "Kayal", "Tanushree",
]
ASR_COLORS = {key: PALETTE[index] for index, key in enumerate(ASR_ENGINES)}

# Which engine may run on which variety. (supported, note shown in the results).
# Mirrors the notebook's routing: unsupported pairs are reported, never run.
_SUPPORT = {
    "koyal": {"hi": (True, ""),
              "hi-en": (True, "Monolingual Hindi model; English words are expected in Devanagari."),
              "sa": (False, "Koyal is a Hindi-only model.")},
    "sarvam-saaras-v3": {"hi": (True, ""), "hi-en": (True, ""),
                         "sa": (True, "No Sanskrit language code; sent as hi-IN.")},
    "whisper-large-v3-turbo": {"hi": (True, ""),
                               "hi-en": (True, "Decoder language not forced for code-mixed speech."),
                               "sa": (True, "Whisper Sanskrit decoder (sa).")},
    "indic-conformer-ctc": {"hi": (True, ""), "hi-en": (True, "Hindi decoder on code-mixed speech."),
                            "sa": (True, "")},
    "indic-conformer-rnnt": {"hi": (True, ""), "hi-en": (True, "Hindi decoder on code-mixed speech."),
                             "sa": (True, "")},
    "sravaani": {"hi": (True, ""), "hi-en": (True, ""), "sa": (True, "")},
    "bulbul-v3": {"hi": (True, ""), "hi-en": (True, ""),
                  "sa": (False, "No configured Sanskrit support in Bulbul v3.")},
    "maya2-native": {"hi": (True, ""), "hi-en": (True, "Maya receives Hindi as the synthesis language; review code-mixed pronunciation."),
                     "sa": (False, "Maya 2 Native does not list Sanskrit among its supported languages.")},
    "mms-tts-hin": {"hi": (True, ""),
                    "hi-en": (True, "Latin-script words may be mispronounced (model limitation)."),
                    "sa": (False, "No validated Sanskrit MMS checkpoint configured.")},
}


def engine_support(key: str, language: str) -> tuple[bool, str]:
    return _SUPPORT.get(key, {}).get(language, (True, ""))
INPUT_LABEL = "Input speech"
TTS_COLORS = {INPUT_LABEL: REFERENCE_COLOR} | {
    TTS_ENGINES[key]["label"]: PALETTE[index] for index, key in enumerate(TTS_ENGINES)
}


# ---------------------------------------------------------------- model loaders


def _inference_device(use_gpu: bool = True) -> torch.device:
    return torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_whisper(device: str = "cpu"):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID,
        revision=WHISPER_REVISION,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device).eval()
    return model, AutoProcessor.from_pretrained(WHISPER_MODEL_ID, revision=WHISPER_REVISION)


@st.cache_resource(show_spinner=False)
def load_indic_conformer(device: str = "cpu"):
    from transformers import AutoModel
    from huggingface_hub import snapshot_download

    token = secret("HF_TOKEN") or None
    snapshot_dir = Path(snapshot_download(
        INDIC_CONFORMER_MODEL_ID,
        revision=INDIC_CONFORMER_REVISION,
        token=token,
    ))
    _materialize_snapshot_files(snapshot_dir)
    return AutoModel.from_pretrained(
        INDIC_CONFORMER_MODEL_ID,
        revision=INDIC_CONFORMER_REVISION,
        trust_remote_code=True,
        token=token,
    ).to(device).eval()


def _materialize_snapshot_files(snapshot_dir: Path) -> None:
    for path in snapshot_dir.rglob("*"):
        if not path.is_symlink():
            continue
        target = path.resolve(strict=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            shutil.copyfile(target, temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


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
def load_sravaani(device: str = "cpu"):
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
        ).to(device).eval()
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
def load_mms(device: str = "cpu"):
    from transformers import VitsModel, VitsTokenizer

    model = VitsModel.from_pretrained(
        MMS_MODEL_ID, revision=MMS_REVISION, torch_dtype=torch.float32
    ).to(device).eval()
    tokenizer = VitsTokenizer.from_pretrained(MMS_MODEL_ID, revision=MMS_REVISION)
    return model, tokenizer


@st.cache_resource(show_spinner="Loading Maya 2 Native on the GPU…")
def load_maya(device: str = "cpu"):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Maya 2 Native currently requires Linux; use WSL2 with GPU passthrough or a Linux deployment.")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Maya 2 Native requires a CUDA-enabled NVIDIA GPU; enable the GPU option and GPU hardware.")
    if torch.version.cuda != "12.8":
        raise RuntimeError(
            f"Maya 2 Native requires the CUDA 12.8 PyTorch build; this environment reports CUDA {torch.version.cuda or 'unavailable'}. "
            "Reinstall dependencies from requirements.txt."
        )
    try:
        from maya3 import Maya3TTS
    except ImportError as exc:
        raise RuntimeError(
            "Maya 2 Native is not installed. Rebuild the app with its CUDA 12.8 / Torch 2.8 dependency profile."
        ) from exc
    return Maya3TTS.from_pretrained("maya-research/maya2-native", max_seats=1)


@st.cache_resource(show_spinner=False)
def load_utmos(device: str = "cpu"):
    return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()


RELEASERS = {
    "koyal": load_koyal,
    "whisper-large-v3-turbo": load_whisper,
    "indic-conformer-ctc": load_indic_conformer,
    "indic-conformer-rnnt": load_indic_conformer,
    "sravaani": load_sravaani,
    "mms-tts-hin": load_mms,
    "maya2-native": load_maya,
}
# Loaders are called in their own timed step, so latency and RTF measure inference only.
LOADERS = RELEASERS


def release_unless_needed(key: str, upcoming: list[str]) -> None:
    """Release a local model unless a later step uses the same loader (CTC and RNNT share one)."""
    loader = RELEASERS.get(key)
    if loader is None or any(RELEASERS.get(k) is loader for k in upcoming):
        return
    release(key)


def release(key: str) -> None:
    loader = RELEASERS.get(key)
    if loader is not None:
        if key == "maya2-native":
            loader("cuda:0").close()
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


def preprocess(y: np.ndarray, sr: int, denoise: bool, clipping_fraction: float | None = None) -> dict:
    """clipping_fraction should come from decode_audio(return_info=True): measured before
    peak scaling. Measured here it would always be 0, because decode scales peaks to 0.95."""
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
        "clipping_fraction": (float(clipping_fraction) if clipping_fraction is not None
                              else float(np.mean(np.abs(y) >= 0.999))),
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
    device = str(_inference_device(options.get("use_gpu", True)))
    return " ".join(nemo_transcribe_segments(load_koyal(device), _segments(y, windows), sr))


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
                # No Sanskrit code is configured; Sanskrit is sent as hi-IN (see engine_support).
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
    device = str(_inference_device(options.get("use_gpu", True)))
    model, processor = load_whisper(device)
    language = options.get("language", "hi")
    # Code-mixed speech: let Whisper pick the language per window instead of forcing Hindi.
    generate_args = {"task": "transcribe", "max_new_tokens": 384}
    if language in ("hi", "sa"):
        generate_args["language"] = language
    texts = []
    for segment in _segments(y, windows):
        features = processor(segment, sampling_rate=sr, return_tensors="pt").input_features.to(
            device=device, dtype=model.dtype
        )
        with torch.inference_mode():
            tokens = model.generate(features, **generate_args)
        texts.append(processor.batch_decode(tokens, skip_special_tokens=True)[0].strip())
    return " ".join(texts)


def _asr_indic_conformer(decoder):
    def run(y, sr, windows, options):
        device = str(_inference_device(options.get("use_gpu", True)))
        model = load_indic_conformer(device)
        language = "sa" if options.get("language") == "sa" else "hi"
        texts = []
        with torch.inference_mode():
            for segment in _segments(y, windows):
                inputs = torch.from_numpy(segment).unsqueeze(0).to(device)
                texts.append(_text_from(model(inputs, language, decoder)))
        return " ".join(texts)

    return run


def asr_sravaani(y, sr, windows, options):
    device = str(_inference_device(options.get("use_gpu", True)))
    model = load_sravaani(device)
    available_mb = None if device.startswith("cuda") else _container_memory_available_mb()
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
    """Pack sentences into chunks of at most max_chars, keeping the original order."""
    sentences = [s.strip() for s in re.split(r"(?<=[।॥!?.])\s+", text) if s.strip()]
    chunks, current = [], ""
    for sentence in sentences:
        if len(sentence) > max_chars and current:
            # Fix: flush the buffered earlier sentence BEFORE emitting pieces of a long one;
            # previously the long sentence's pieces jumped ahead of it.
            chunks.append(current)
            current = ""
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


def tts_bulbul(text, language="hi"):
    wav, elapsed, _ = synthesize(text)
    audio, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=False)
    return np.asarray(audio, np.float32), int(sr), elapsed


def tts_maya(text, language="hi", voice="Ahana"):
    """Generate speech locally with the cached Maya 2 Native GPU runtime."""
    model = load_maya("cuda:0")
    started = time.perf_counter()
    try:
        audio = model.generate(
            text,
            lang="hi" if language in {"hi", "hi-en"} else language,
            voice=voice,
            normalize=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Maya 2 Native local generation failed: {type(exc).__name__}: {exc}") from exc
    return np.asarray(audio, np.float32), 24_000, time.perf_counter() - started


def tts_mms(text, language="hi", device="cpu"):
    model, tokenizer = load_mms(device)
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
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            parts += [model(**inputs).waveform[0].float().cpu().numpy(), silence]
    if not parts:
        raise RuntimeError("MMS-TTS produced no audio for this text.")
    return np.concatenate(parts[:-1]).astype(np.float32), sr, time.perf_counter() - started


TTS_RUNNERS = {"bulbul-v3": tts_bulbul, "mms-tts-hin": tts_mms}


def predicted_utmos(audio: np.ndarray, sr: int, device: str = "cpu") -> float:
    model = load_utmos(device)
    a = librosa.resample(audio, orig_sr=sr, target_sr=16000) if sr != 16000 else audio
    a = np.ascontiguousarray(a, dtype=np.float32)
    window, scores, weights = 10 * 16000, [], []
    with torch.inference_mode():
        for start in range(0, len(a), window):
            chunk = a[start:start + window]
            samples = len(chunk)
            if samples < 16000:
                chunk = np.pad(chunk, (0, 16000 - samples))
            scores.append(float(model(torch.from_numpy(chunk).unsqueeze(0).to(device), 16000)))
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


def compare_prosody(reference: dict, candidate: dict, truncated: bool = False) -> dict:
    """truncated=True when the TTS text was cut to TTS_COMPARISON_MAX_CHARS: the generated
    speech then covers less content than the input, so the duration ratio is not reported."""
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
        "Duration ratio": float("nan") if truncated else ratio("Duration (s)"),
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
    device = str(_inference_device(options.get("use_gpu", True)))
    options = options | {"device": device}
    status.write(f"Local model inference device: {device}. Audio preprocessing and scoring remain on CPU.")
    log.append({"Step": "Compute device", "Status": "ok", "Seconds": 0.0, "Detail": device})

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

    language = options.get("language", "hi")
    judge = options["judge"]
    pre, error = step("Preprocessing (normalize, VAD, denoise)", preprocess, audio, sr, options["denoise"],
                      options.get("clipping_fraction"))
    if error:
        raise RuntimeError(f"Preprocessing failed: {error}")
    duration = len(audio) / sr

    def load_model(key, label):
        """Load in its own timed step so latency/RTF below measure inference only."""
        loader = LOADERS.get(key)
        if loader is None:
            return 0.0, None
        started = time.perf_counter()
        _, error = step(f"Load · {label}", loader, device)
        return time.perf_counter() - started, error

    asr_rows = []
    for index, key in enumerate(options["asr"]):
        label = ASR_ENGINES[key]["label"]
        supported, note = engine_support(key, language)
        row = {"key": key, "label": label, "note": note, "text": "", "error": None,
               "latency": float("nan"), "rtf": float("nan"), "load_seconds": float("nan")}
        if not supported:
            row |= {"status": "unsupported", "error": note}
            log.append({"Step": f"ASR · {label}", "Status": "unsupported", "Seconds": 0.0, "Detail": note})
            asr_rows.append(row)
            continue
        row["load_seconds"], error = load_model(key, label)
        text = None
        if not error:
            started = time.perf_counter()
            text, error = step(f"ASR · {label}", ASR_RUNNERS[key], pre["processed"], sr, pre["windows"], options)
            elapsed = time.perf_counter() - started
            row |= {"latency": elapsed, "rtf": elapsed / duration}
        row |= {"status": "failed" if error else "ok", "error": error, "text": text or ""}
        if not error:
            if not row["text"]:
                row |= {"status": "failed", "error": "Model returned an empty transcript."}
            else:
                row |= score_transcript(reference_text, row["text"], language)
        asr_rows.append(row)
        if options["release"]:
            release_unless_needed(key, options["asr"][index + 1:] + ([judge] if judge else []))

    tts_text, source = reference_text, "Reference transcript"
    if options["tts_source"] == "custom":
        tts_text = str(options.get("custom_tts_text", "")).strip()
        source = "Reviewed custom text"
    elif options["tts_source"] != "reference":
        source_row = next((r for r in asr_rows if r["key"] == options["tts_source"] and r["status"] == "ok"), None)
        if source_row:
            tts_text, source = source_row["text"], f"{source_row['label']} transcript"
        else:
            log.append({"Step": "TTS text", "Status": "fallback", "Seconds": 0.0,
                        "Detail": "Selected ASR transcript unavailable; using the reference transcript."})
    tts_text, truncated = sentence_safe_prefix(tts_text, TTS_COMPARISON_MAX_CHARS)

    tts_rows = []
    for index, key in enumerate(options["tts"]):
        label = TTS_ENGINES[key]["label"]
        supported, note = engine_support(key, language)
        if not supported:
            tts_rows.append({"key": key, "label": label, "status": "unsupported", "error": note, "note": note})
            log.append({"Step": f"TTS · {label}", "Status": "unsupported", "Seconds": 0.0, "Detail": note})
            continue
        load_seconds, error = load_model(key, label)
        result = None
        if not error:
            if key == "maya2-native":
                result, error = step(
                    f"TTS · {label}",
                    tts_maya,
                    tts_text,
                    language,
                    options.get("maya_voice", "Ahana"),
                )
            elif key == "mms-tts-hin":
                result, error = step(f"TTS · {label}", TTS_RUNNERS[key], tts_text, language, device)
            else:
                result, error = step(f"TTS · {label}", TTS_RUNNERS[key], tts_text, language)
        row = {"key": key, "label": label, "note": note, "load_seconds": load_seconds,
               "status": "failed" if error else "ok", "error": error}
        if key == "maya2-native":
            row["voice"] = options.get("maya_voice", "Ahana")
        if result:
            out, out_sr, elapsed = result
            out_duration = len(out) / out_sr
            row |= {"audio": out, "sr": out_sr, "wav": _wav_bytes(out, out_sr), "latency": elapsed,
                    "duration": out_duration, "rtf": elapsed / out_duration if out_duration else float("nan")}
            if not np.isfinite(out).all() or not len(out):
                row |= {"status": "failed", "error": "Model returned empty or invalid audio."}
        tts_rows.append(row)
        if options["release"] and not (
            key == "maya2-native" and options.get("defer_maya_release")
        ):
            release_unless_needed(key, options["tts"][index + 1:])

    judge_ok = bool(judge) and engine_support(judge, language)[0]
    if judge and not judge_ok:
        log.append({"Step": "Loopback", "Status": "unsupported", "Seconds": 0.0,
                    "Detail": f"{ASR_ENGINES[judge]['label']} does not support {LANGUAGES.get(language, language)}."})
    for row in (r for r in tts_rows if r["status"] == "ok"):
        speech = normalize_rms(_to_16k(row["audio"], row["sr"]))
        row["vad"], _ = step(f"VAD · {row['label']} output", run_vad, speech, SAMPLE_RATE)
        row["audio16"] = speech
        if judge_ok:
            text, error = step(f"Loopback · {row['label']} → {ASR_ENGINES[judge]['label']}",
                               ASR_RUNNERS[judge], speech, SAMPLE_RATE,
                               build_windows(len(speech), SAMPLE_RATE, row["vad"] or []), options)
            row["loopback_text"] = text or ""
            row["loopback_error"] = error
            if text:
                row["loopback"] = score_transcript(tts_text, text, language)
        if options["utmos"]:
            row["utmos"], row["utmos_error"] = step(
                f"UTMOS · {row['label']}", predicted_utmos, row["audio"], row["sr"], device
            )
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

    synthetic = None
    if options.get("deepfake"):
        import deepfake

        clips = [(INPUT_LABEL, "input (provenance unknown)", pre["normalized"], sr)]
        clips += [(r["label"], "generated (known synthetic)", r["audio"], r["sr"])
                  for r in tts_rows if r["status"] == "ok"]
        synthetic, _ = step("Synthetic-speech check", deepfake.score_clips, clips,
                            st.session_state.get("deepfake_detector"), device)

    return {"audio": audio, "sr": sr, "reference": reference_text, "language": language, "pre": pre,
            "asr": asr_rows, "tts_text": tts_text, "tts_source": source, "tts_truncated": truncated,
            "tts": tts_rows, "judge": ASR_ENGINES[judge]["label"] if judge_ok else None,
            "prosody": prosody, "synthetic": synthetic, "log": log}


# ---------------------------------------------------------------- charts


def _bar(frame: pd.DataFrame, field: str, title: str, colors: dict, color_field: str, fmt: str = ".2f", rule: float | None = None):
    frame = frame.dropna(subset=[field])
    if frame.empty:
        return None
    order = [k for k in colors if k in set(frame[color_field])]
    bars = alt.Chart(frame, title=title).mark_bar(cornerRadiusEnd=4, height={"band": 0.6}).encode(
        x=alt.X(f"{field}:Q", title=None),
        y=alt.Y(
            f"{color_field}:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelLimit=240, labelFontSize=12),
            scale=alt.Scale(paddingInner=0.4),
        ),
        color=alt.Color(f"{color_field}:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())), legend=None),
        tooltip=[alt.Tooltip(f"{color_field}:N", title="Model"), alt.Tooltip(f"{field}:Q", title=title, format=fmt)],
    )
    labels = bars.mark_text(align="left", dx=4, color="#52514e").encode(text=alt.Text(f"{field}:Q", format=fmt), color=alt.value("#52514e"))
    chart = bars + labels
    if rule is not None:
        chart += alt.Chart(pd.DataFrame({"x": [rule]})).mark_rule(strokeDash=[4, 4], color="#8a8985").encode(x="x:Q")
    return chart.properties(height=max(150, 56 * len(frame)))


def _show(chart):
    if chart is not None:
        st.altair_chart(chart, width="stretch")


def _fig_close(fig, alt_text):
    try:
        image = io.BytesIO()
        fig.savefig(image, format="png", bbox_inches="tight")
        st.image(media_data_uri(image.getvalue(), "image/png"), caption=alt_text, width="stretch")
    finally:
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
    language = result.get("language", "hi")
    primary = {"hi": "wer", "hi-en": "wer_script_tolerant", "sa": "cer"}.get(language, "wer")
    st.caption(
        f"Language: **{LANGUAGES.get(language, language)}** · headline metric: **{PRIMARY_LABEL[primary]}**. "
        "Normalized = numbers to words, nukta and chandrabindu folding, acronyms, loanwords (the notebook's "
        "scoring). Raw = punctuation and case only. Script-tolerant also accepts the same word written in "
        "Latin and Devanagari. Latency and RTF exclude model loading (shown separately)."
    )
    table = pd.DataFrame([{
        "Model": r["label"], "Status": r["status"],
        "WER norm (%)": r.get("wer"), "WER raw (%)": r.get("wer_raw"),
        "WER script-tol. (%)": r.get("wer_script_tolerant"), "CER norm (%)": r.get("cer"),
        "Subs": r.get("substitutions"), "Dels": r.get("deletions"), "Ins": r.get("insertions"),
        "Latency (s)": r["latency"], "RTF": r["rtf"], "Load (s)": r.get("load_seconds"),
        "Note": r.get("note", ""),
    } for r in rows])
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.2f")
        for c in ("WER norm (%)", "WER raw (%)", "WER script-tol. (%)", "CER norm (%)", "Latency (s)", "RTF", "Load (s)")
    })
    for r in rows:
        if r["status"] == "unsupported":
            st.info(f"**{r['label']}** not run — {r['error']}")
        elif r["status"] != "ok":
            st.error(f"**{r['label']}** failed — {r['error']}")
    if not ok:
        return

    frame = pd.DataFrame([{"Model": r["label"], "Primary": r["primary"], "WER": r["wer"], "CER": r["cer"],
                           "Latency": r["latency"], "RTF": r["rtf"]} for r in ok])
    colors = {ASR_ENGINES[k]["label"]: c for k, c in ASR_COLORS.items()}
    _show(_bar(frame, "Primary", f"{PRIMARY_LABEL[primary]} (%) — lower is better", colors, "Model"))
    if primary != "cer":
        _show(_bar(frame, "CER", "Character error rate (%) — lower is better", colors, "Model"))

    errors = pd.DataFrame([
        {"Model": r["label"], "Error type": kind, "Count": r[field]}
        for r in ok for kind, field in (("Substitution", "substitutions"), ("Deletion", "deletions"), ("Insertion", "insertions"))
    ])
    kinds = ["Substitution", "Deletion", "Insertion"]
    stacked = alt.Chart(errors, title="Word errors by type").mark_bar(stroke="white", strokeWidth=2, height={"band": 0.6}).encode(
        x=alt.X("Count:Q", stack="zero", title="Word errors"),
        y=alt.Y(
            "Model:N",
            sort=[l for l in colors if l in set(errors["Model"])],
            title=None,
            axis=alt.Axis(labelLimit=240, labelFontSize=12),
            scale=alt.Scale(paddingInner=0.4),
        ),
        color=alt.Color("Error type:N", scale=alt.Scale(domain=kinds, range=PALETTE[:3]), legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("order:Q"),
        tooltip=["Model", "Error type", "Count"],
    ).transform_calculate(order="indexof(['Substitution','Deletion','Insertion'], datum['Error type'])")
    _show(stacked.properties(height=max(150, 56 * len(ok))))

    _show(_bar(frame, "Latency", "Processing time (s)", colors, "Model"))
    _show(_bar(frame, "RTF", "Real-time factor — below 1 is faster than real time", colors, "Model", fmt=".3f", rule=1.0))

    st.markdown("#### Word-level alignment against the reference")
    st.caption("Highlighted = substitution (reference in brackets) · strike-through = deleted reference word · underline = inserted word.")
    for r in ok:
        with st.expander(f"{r['label']} — {PRIMARY_LABEL[primary]} {r['primary']:.1f}%"):
            st.markdown(_alignment_html(r["alignment"]), unsafe_allow_html=True)
            st.caption("Raw transcript")
            st.text(r["text"])


def render_tts(result):
    st.markdown(f"**Text sent to every TTS model** ({result['tts_source']}"
                f"{', trimmed at a sentence boundary to ' + str(TTS_COMPARISON_MAX_CHARS) + ' characters' if result['tts_truncated'] else ''}):")
    st.text(result["tts_text"])
    rows = result["tts"]
    st.audio(media_data_uri(_wav_bytes(result["audio"], result["sr"]), "audio/wav"), format="audio/wav")
    st.caption("Original input speech")
    for r in rows:
        if r["status"] == "ok":
            model_label = f"{r['label']} · {r['voice']}" if r.get("voice") else r["label"]
            st.markdown(f"**{model_label}**")
            st.audio(media_data_uri(r["wav"], "audio/wav"), format="audio/wav")
        elif r["status"] == "unsupported":
            st.info(f"**{r['label']}** not run — {r['error']}")
        else:
            st.error(f"**{r['label']}** failed — {r['error']}")
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return
    judge = result["judge"]
    table = pd.DataFrame([{
        "Model": f"{r['label']} · {r['voice']}" if r.get("voice") else r["label"],
        "Duration (s)": r["duration"], "Latency (s)": r["latency"], "RTF": r["rtf"],
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

    render_voice_control(result)

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


def render_voice_control(result):
    """Rank TTS candidates using median F0 as a rough voice-range heuristic."""
    ok = [r for r in result["tts"] if r["status"] == "ok"]
    if not ok:
        return
    st.markdown("#### Voice control — automatically rank higher-pitch candidates")
    st.caption(
        f"Candidates with median F0 at or above {FEMALE_VOICE_F0_THRESHOLD_HZ:.0f} Hz are "
        "included automatically. F0 is a rough voice-range heuristic, not a gender test."
    )
    prosody = result.get("prosody", {})
    ranked = []
    for r in ok:
        raw_f0 = prosody.get(r["label"], {}).get("metrics", {}).get("F0 median (Hz)")
        try:
            f0 = float(raw_f0)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f0) and f0 >= FEMALE_VOICE_F0_THRESHOLD_HZ:
            ranked.append((r, f0))
    if not ranked:
        st.info(
            "No successful TTS output has a measurable median F0 at or above "
            f"{FEMALE_VOICE_F0_THRESHOLD_HZ:.0f} Hz, so no ranking is shown."
        )
        return
    ranking = pd.DataFrame([{
        "Model": r["label"],
        "Median F0 (Hz)": f0,
        "Loopback WER (%)": r.get("loopback", {}).get("primary", np.nan),
        "Predicted UTMOS": r.get("utmos") if r.get("utmos") is not None else np.nan,
        "RTF": r["rtf"],
    } for r, f0 in ranked]).sort_values(["Loopback WER (%)", "Predicted UTMOS"], ascending=[True, False])
    st.dataframe(ranking, hide_index=True, width="stretch",
                 alt="TTS candidates with median F0 at or above the heuristic threshold, ranked by quality metrics",
                 column_config={
        c: st.column_config.NumberColumn(format="%.2f") for c in ranking.columns if c != "Model"
    })
    st.caption("Ranked by loopback error (intelligibility proxy), then predicted UTMOS.")


def render_synthetic(result):
    import deepfake

    deepfake.render_scores(result.get("synthetic"))


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
            label: compare_prosody(prosody[INPUT_LABEL], analysis, result.get("tts_truncated", False))
            for label, analysis in prosody.items() if label != INPUT_LABEL
        })
        comparison.index.name = "Compared with input speech"
        st.markdown("#### How closely each TTS voice follows the input speaker")
        st.dataframe(comparison, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in comparison.columns})
        st.caption("Correlations compare normalized contour shapes over relative time (1 = same shape). Ratios near 1 and |diff| near 0 mean closer to the input."
                   + (" Duration ratio is blank because the TTS text was trimmed, so the generated speech covers less content than the input."
                      if result.get("tts_truncated") else "")
                   + " This is descriptive resemblance, not word-aligned fidelity or voice cloning.")

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


def _clear_research_result() -> None:
    st.session_state.pop("research_result", None)
    st.session_state.pop("research_result_digest", None)


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
        recording = st.audio_input(
            "Record Hindi speech",
            key="research_recording",
            on_change=_clear_research_result,
        )
    with right:
        upload = st.file_uploader(
            "Or upload audio",
            type=["wav", "mp3", "m4a", "mp4", "flac", "ogg", "aac"],
            key="research_upload",
            on_change=_clear_research_result,
        )
    source = recording or upload
    source_bytes = source.getvalue() if source is not None else None
    audio_quality = None
    if source_bytes is not None:
        try:
            audio_quality = audio_quality_preview(
                source_bytes,
                Path(getattr(source, "name", "recording.wav")).suffix,
                MAX_RESEARCH_SECONDS,
            )
            st.caption(
                f"Audio check · {audio_quality['duration']:.1f} s · "
                f"peak {audio_quality['peak']:.3f} · RMS {audio_quality['rms_dbfs']:.1f} dBFS"
            )
            if audio_quality["rms_dbfs"] < -35:
                st.warning("The recording is very quiet; try a closer microphone position before analysis.")
            if audio_quality["clipping_fraction"] >= 0.005:
                st.warning("The recording may be clipped; reduce the input level before analysis.")
        except Exception as exc:
            st.error(f"Could not inspect the recording: {type(exc).__name__}: {exc}")

    # Suggest the language from the reference text: preset the selector once per new reference,
    # so the user can still override it afterwards.
    suggestion, evidence = guess_language(reference_text) if reference_ok else (None, "")
    reference_key = hashlib.sha256(reference_text.encode("utf-8")).hexdigest() if reference_ok else None
    if suggestion and st.session_state.get("research_language_ref") != reference_key:
        st.session_state["research_language"] = suggestion
        st.session_state["research_language_ref"] = reference_key

    with st.container(border=True):
        st.markdown("**Models**")
        if suggestion:
            st.caption(f"Suggested from the reference: **{LANGUAGES[suggestion]}** — {evidence}.")
        language = st.selectbox(
            "Language variety", list(LANGUAGES), format_func=LANGUAGES.get, key="research_language",
            help="Routes each model to its language setting and picks the headline metric: "
                 "normalized WER (Hindi), script-tolerant WER (code-mixed), CER (Sanskrit). "
                 "Unsupported model-language pairs are reported, not run.",
        )
        asr = st.multiselect("ASR models", list(ASR_ENGINES), default=["koyal", "sarvam-saaras-v3"],
                             format_func=_format_engine(ASR_ENGINES), key="research_asr")
        default_tts = ["bulbul-v3", "mms-tts-hin"]
        tts = st.multiselect("TTS models", list(TTS_ENGINES), default=default_tts,
                             format_func=_format_engine(TTS_ENGINES), key="research_tts")
        if "maya2-native" in tts:
            if st.session_state.get("research_maya_voice") not in MAYA_FEMALE_VOICES:
                st.session_state["research_maya_voice"] = "Ahana"
            maya_voice = st.selectbox(
                "Maya female voice",
                MAYA_FEMALE_VOICES,
                key="research_maya_voice",
                help="Only Maya's documented female voice personas are offered here.",
            )
            st.caption(
                "Maya runs locally with Maya3TTS; no endpoint URL or API key is needed. "
                "First use downloads the model and requires about 9.2 GB VRAM at peak; 12 GB or more is recommended."
            )
        else:
            maya_voice = "Ahana"
        columns = st.columns(2)
        tts_source = columns[0].selectbox(
            "Text for TTS", ["reference", "custom", *asr],
            format_func=lambda k: (
                "Reference transcript" if k == "reference"
                else "Reviewed custom text" if k == "custom"
                else f"{ASR_ENGINES[k]['label']} transcript"
            ),
            key="research_tts_source",
        )
        judge = columns[1].selectbox(
            "Loopback judge (re-transcribes TTS output)", [None, *asr],
            index=1 if asr else 0,
            format_func=lambda k: "None" if k is None else ASR_ENGINES[k]["label"],
            key="research_judge",
        )
        gpu_available = torch.cuda.is_available()
        columns = st.columns(6)
        sarvam_mode = columns[0].selectbox("Saaras mode", ["codemix", "transcribe"], key="research_sarvam_mode")
        denoise = columns[1].toggle("Adaptive denoise", value=True, key="research_denoise")
        utmos = columns[2].toggle("UTMOS score", value=True, key="research_utmos", help="Downloads the SpeechMOS UTMOS22 model on first use.")
        release_models = columns[3].toggle("Free models after use", value=True, key="research_release",
                                           help="Unloads local models after each ASR/TTS step to reduce peak memory. A selected loopback judge may be loaded again later. Loading time is reported separately from inference.")
        deepfake_check = columns[4].toggle("Synthetic-speech check", value=False, key="research_deepfake",
                                           help="Scores the input and every TTS output with the zero-shot Indic detector, plus your own trained detector if one is loaded in Synthetic speech mode.")
        use_gpu = columns[5].toggle(
            "Use GPU for local models",
            value=gpu_available,
            disabled=not gpu_available,
            key="research_use_gpu",
            help="Run supported local ASR/TTS models and UTMOS on CUDA. CPU preprocessing and scoring remain unchanged.",
        )
        if gpu_available and use_gpu:
            gpu = torch.cuda.get_device_properties(0)
            st.caption(f"CUDA enabled · {torch.cuda.get_device_name(0)} · {gpu.total_memory / 1024**3:.1f} GB VRAM")
        elif gpu_available:
            st.caption("Local models will run on CPU; CUDA is available but disabled.")
        else:
            st.info("CUDA GPU not detected. Local models will use CPU. Enable GPU hardware and install a CUDA-enabled PyTorch build to use acceleration.")
        if "maya2-native" in tts and not (gpu_available and use_gpu):
            st.warning("Maya 2 Native requires a CUDA-enabled NVIDIA GPU and the GPU option enabled.")

        if tts_source == "custom":
            custom_tts_text = st.text_area(
                "Review or edit text for TTS",
                value=reference_text,
                height=100,
                key=f"research_custom_tts_{reference_key or 'empty'}",
                help="This text is sent to every selected TTS model instead of the reference or an ASR result.",
            ).strip()
        else:
            custom_tts_text = ""

        unsupported = [ASR_ENGINES[k]["label"] for k in asr if not engine_support(k, language)[0]]
        unsupported += [TTS_ENGINES[k]["label"] for k in tts if not engine_support(k, language)[0]]
        if unsupported:
            st.info(f"Not run for {LANGUAGES[language]}: {', '.join(unsupported)}.")
        needs_hf = {"koyal", "indic-conformer-ctc", "indic-conformer-rnnt", "sravaani"}
        needs_sarvam = {"sarvam-saaras-v3", "bulbul-v3"}
        missing = []
        if needs_hf & set(asr + tts) and missing_secrets(["HF_TOKEN"]):
            missing.append("HF_TOKEN (gated Hugging Face models)")
        if needs_sarvam & set(asr + tts) and missing_secrets(["SARVAM_API_KEY"]):
            missing.append("SARVAM_API_KEY (Sarvam Saaras/Bulbul)")
        if missing:
            st.warning("Missing secrets: " + "; ".join(missing) + ". Those models will fail until the secret is added.")

        heavy = [ASR_ENGINES[k]["label"] for k in asr if ASR_ENGINES[k]["heavy"]] + [TTS_ENGINES[k]["label"] for k in tts if TTS_ENGINES[k]["heavy"]]
        if heavy:
            device_label = "GPU" if gpu_available and use_gpu else "CPU"
            st.warning(
                f"{', '.join(heavy)} run locally on the {device_label}. Expect first-run downloads of several GB. "
                "Check the memory estimates shown beside each model against available RAM/VRAM; deselect "
                "models if you run out of memory."
            )

    if source is None:
        st.info(f"Record or upload a Hindi utterance up to {MAX_RESEARCH_SECONDS} seconds.")
    elif not reference_ok:
        st.warning("Add the golden-reference transcript to enable scoring.")

    options = {"asr": asr, "tts": tts, "tts_source": tts_source, "custom_tts_text": custom_tts_text,
               "maya_voice": maya_voice, "judge": judge, "sarvam_mode": sarvam_mode,
               "denoise": denoise, "utmos": utmos, "release": release_models, "language": language,
               "deepfake": deepfake_check, "use_gpu": use_gpu}
    # Results belong to one exact combination of audio, reference and settings. Any change
    # clears them (previously only an audio change did).
    run_digest = _settings_digest(source_bytes, reference_text, options)
    if st.session_state.get("research_result_digest") != run_digest:
        _clear_research_result()

    can_run = (
        source is not None
        and audio_quality is not None
        and reference_ok
        and bool(asr)
        and (tts_source != "custom" or bool(normalize_reference_text(custom_tts_text)))
    )
    if st.button("Run research comparison", type="primary", width="stretch", disabled=not can_run, key="research_run"):
        _clear_research_result()
        try:
            audio, sr, _, info = decode_audio(source_bytes, Path(getattr(source, "name", "recording.wav")).suffix,
                                              MAX_RESEARCH_SECONDS, return_info=True)
            run_options = options | {"clipping_fraction": info["clipping_fraction"]}
            with st.status("Running research comparison…", expanded=True) as status:
                result = run_research(audio, sr, reference_text, run_options, status)
                result["name"] = getattr(source, "name", "recording")
                result["run_id"] = run_digest[:12]
                st.session_state["research_result"] = result
                st.session_state["research_result_digest"] = run_digest
                status.update(label="Research comparison complete", state="complete", expanded=False)
        except Exception as exc:
            st.error(f"Research run failed: {type(exc).__name__}: {exc}")

    result = st.session_state.get("research_result")
    if not result:
        return
    st.divider()
    st.markdown(f"### Results — {html.escape(str(result['name']))}")
    render_result_tabs(result)


def render_result_tabs(result, log_tab=True):
    names = ["Audio preprocessing", "ASR comparison", "TTS comparison", "Prosody & voice"]
    if result.get("synthetic") is not None:
        names.append("Synthetic speech")
    if log_tab:
        names.append("Run log")
    sections = dict(zip(names, st.tabs(names)))
    with sections["Audio preprocessing"]:
        render_preprocessing(result)
    with sections["ASR comparison"]:
        render_asr(result)
    with sections["TTS comparison"]:
        render_tts(result)
    with sections["Prosody & voice"]:
        render_prosody(result)
    if "Synthetic speech" in sections:
        with sections["Synthetic speech"]:
            render_synthetic(result)
    if log_tab:
        with sections["Run log"]:
            st.dataframe(pd.DataFrame(result["log"]), hide_index=True, width="stretch",
                         column_config={"Seconds": st.column_config.NumberColumn(format="%.2f")})


def _settings_digest(source_bytes, reference_text, options) -> str:
    import json

    digest = hashlib.sha256(source_bytes or b"")
    digest.update((reference_text or "").encode("utf-8"))
    digest.update(json.dumps(options, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()
