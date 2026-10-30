"""Zero-shot voice cloning backends for the naturalness pipeline.

Cloning models pin dependencies that conflict with this app (Maya pins transformers 5.6.2), so
they normally run in their own virtual environment. Set VOICE_CLONE_PYTHON to that environment's
interpreter; this file is then executed there as a worker (`python voice_clone.py jobs.json`).
It imports nothing from the app, so only the backend's own packages are needed in that venv.
Without VOICE_CLONE_PYTHON the backend runs in the current process.

The speaker reference is ONLY the speaker's clone_ref prompt/passage, never a stimulus recording;
naturalness.generate() refuses to clone when the corpus check finds leakage.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

INDICF5_REVISION = "fb652a96e508091045890dae5bb99fe511facbcf"

BACKENDS = {
    "indicf5": {
        "label": "IndicF5 (AI4Bharat, Hindi)",
        "install": "pip install git+https://github.com/ai4bharat/IndicF5.git soundfile",
        "note": "Conditions on a 6-12 s prompt plus its exact transcript. The Hugging Face repo may be gated: "
                "accept its terms and set HF_TOKEN. Check the model licence before use.",
    },
    "xtts": {
        "label": "XTTS-v2 (Coqui, multilingual incl. Hindi)",
        "install": "pip install coqui-tts soundfile",
        "note": "Conditions on the reference passage audio. Coqui Public Model License (non-commercial); "
                "set COQUI_TOS_AGREED=1 after reading it.",
    },
}


def _device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _indicf5(spec: dict):
    import numpy as np
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        "ai4bharat/IndicF5",
        revision=INDICF5_REVISION,
        trust_remote_code=True,
    )
    if spec["device"].startswith("cuda"):
        model = model.to(spec["device"])

    def synth(text: str):
        audio = np.asarray(model(text, ref_audio_path=spec["prompt_wav"], ref_text=spec["prompt_text"]))
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        return audio.astype(np.float32), 24_000

    return synth


def _xtts(spec: dict):
    if os.getenv("COQUI_TOS_AGREED") != "1":
        raise RuntimeError(
            "XTTS-v2 requires COQUI_TOS_AGREED=1 after you have read and accepted the Coqui Public Model License."
        )
    import numpy as np
    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(spec["device"])
    reference = spec.get("passage_wav") or spec["prompt_wav"]

    def synth(text: str):
        wav = tts.tts(text=text, speaker_wav=reference, language="hi")
        return np.asarray(wav, np.float32), int(tts.synthesizer.output_sample_rate)

    return synth


LOADERS = {"indicf5": _indicf5, "xtts": _xtts}


def run_jobs(spec: dict, emit) -> int:
    """Load the backend once and synthesize every job. emit(dict) reports progress per file."""
    import soundfile as sf

    spec = spec | {"device": _device(spec.get("device"))}
    started = time.perf_counter()
    synth = LOADERS[spec["backend"]](spec)
    emit({"event": "loaded", "backend": spec["backend"], "device": spec["device"],
          "seconds": round(time.perf_counter() - started, 1)})
    done, failures = 0, []
    for job in spec["jobs"]:
        try:
            audio, sr = synth(job["text"])
            Path(job["out"]).parent.mkdir(parents=True, exist_ok=True)
            sf.write(job["out"], audio, sr, subtype="PCM_16")
            done += 1
            emit({"event": "done", "out": job["out"]})
        except Exception as exc:  # keep going; one bad sentence should not lose the batch
            error = f"{type(exc).__name__}: {exc}"
            failures.append((job["out"], error))
            emit({"event": "error", "out": job["out"], "error": error})
    if failures:
        details = "; ".join(f"{out}: {error}" for out, error in failures[:10])
        extra = f" (+{len(failures) - 10} more)" if len(failures) > 10 else ""
        raise RuntimeError(f"Voice cloning failed for {len(failures)} of {len(spec['jobs'])} file(s): {details}{extra}")
    return done


def generate_batch(backend: str, prompt_wav: str, prompt_text: str, jobs: list[dict],
                   passage_wav: str | None = None, device: str | None = None, log=print) -> int:
    if backend not in LOADERS:
        raise ValueError(f"Unknown cloning backend {backend!r}; choose from {sorted(LOADERS)}")
    spec = {"backend": backend, "prompt_wav": prompt_wav, "prompt_text": prompt_text,
            "passage_wav": passage_wav, "device": device, "jobs": jobs}

    def emit(event: dict):
        if event["event"] == "error":
            log(f"clone error {event['out']}: {event['error']}")
        elif event["event"] == "loaded":
            log(f"{backend} loaded on {event['device']} in {event['seconds']} s")
        else:
            log(f"cloned {Path(event['out']).name}")

    python = os.getenv("VOICE_CLONE_PYTHON")
    if not python:
        try:
            return run_jobs(spec, emit)
        except ImportError as exc:
            info = BACKENDS[backend]
            raise RuntimeError(f"{info['label']} is not installed here ({exc}). Create a separate venv, run "
                               f"`{info['install']}`, and set VOICE_CLONE_PYTHON to its python.") from exc
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(spec, handle, ensure_ascii=False)
        spec_path = handle.name
    try:
        proc = subprocess.Popen(
            [python, str(Path(__file__).resolve()), spec_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        done = 0
        output_tail = deque(maxlen=30)
        if proc.stdout is None:
            raise RuntimeError("Could not read output from the voice-cloning worker.")
        for line in proc.stdout:
            output_tail.append(line.rstrip())
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            done += event["event"] == "done"
            emit(event)
        return_code = proc.wait()
        if return_code:
            raise RuntimeError(f"Cloning worker failed ({python}, exit {return_code}):\n" + "\n".join(output_tail))
        return done
    finally:
        Path(spec_path).unlink(missing_ok=True)


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        job_spec = json.load(f)
    run_jobs(job_spec, lambda event: print(json.dumps(event, ensure_ascii=False), flush=True))
