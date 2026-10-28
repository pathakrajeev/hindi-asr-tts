"""Synthetic-speech (deepfake) detection for the Hindi ASR/TTS Streamlit app.

Two detectors, both reported as research evidence, never as proof of authenticity:

1. Zero-shot baseline: a pretrained Indic TTS detector (DistilHuBERT, 2 s windows, label 1 = TTS),
   trained by its author on the IndicTTS Deepfake Challenge data. Its reported accuracy is on its
   own validation split only.
2. Own detector: mean-pooled WavLM embeddings + standardized logistic regression, trained on real
   recordings vs this app's Bulbul / MMS / Parler outputs (and any labelled files you add).

Evaluation follows two protocols so a high score cannot come from memorizing a voice or sentence:
  * speaker/content-disjoint: GroupKFold, a group = speaker or source recording;
  * unseen generator: train without one TTS engine, test on it against held-out real speech.
Every clip gets the same loudness normalization and (optionally) the same MP3 round-trip, so
level or codec differences cannot give the answer away.
"""
from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

ZERO_SHOT_ID = "Khon198/indictts-deepfake-detector"
ZERO_SHOT_REVISION = None  # pin a commit hash here before reporting thesis numbers
EMBED_ID = "microsoft/wavlm-base-plus"
EMBED_REVISION = None      # pin a commit hash here before reporting thesis numbers
SR = 16_000
WINDOW = 2 * SR            # the zero-shot model was trained on 2-second windows
SYNTHETIC_INDEX = 1        # model card: is_tts = 1 means AI-generated; checked against id2label at load
MIN_REAL_GROUPS_FOR_CLAIMS = 30
TARGET_RMS_DBFS = -20.0
AUDIO_TYPES = ["wav", "mp3", "m4a", "mp4", "flac", "ogg", "aac"]


# ---------------------------------------------------------------- audio preparation


def prepare(audio: np.ndarray, sr: int) -> np.ndarray:
    """16 kHz mono float32 at -20 dBFS RMS (the same level for real and generated clips)."""
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64)) + 1e-12))
    y = y * (10 ** (TARGET_RMS_DBFS / 20) / max(rms, 1e-8))
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.89:
        y = y / peak * 0.89
    return y.astype(np.float32)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def mp3_roundtrip(y: np.ndarray, sr: int = SR, bitrate: str = "64k") -> np.ndarray:
    """Codec control: real and generated clips get the same MP3 encode and decode."""
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is not installed, so the MP3 codec control cannot run.")
    with tempfile.TemporaryDirectory(prefix="deepfake_codec_") as folder:
        source, target = Path(folder) / "in.wav", Path(folder) / "out.mp3"
        sf.write(source, y, sr, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-b:a", bitrate, str(target)],
                       check=True, timeout=120)
        out, _ = librosa.load(target, sr=sr, mono=True)
    return out.astype(np.float32)


def decode_upload(raw: bytes, suffix: str) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=suffix or ".wav", delete=False) as file:
        file.write(raw)
        path = file.name
    try:
        audio, sr = librosa.load(path, sr=SR, mono=True)
    finally:
        Path(path).unlink(missing_ok=True)
    if not len(audio):
        raise ValueError("Audio file is empty.")
    return audio.astype(np.float32), SR


# ---------------------------------------------------------------- zero-shot detector


@st.cache_resource(show_spinner="Loading the zero-shot synthetic-speech detector…")
def load_zero_shot():
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    kwargs = {"revision": ZERO_SHOT_REVISION} if ZERO_SHOT_REVISION else {}
    extractor = AutoFeatureExtractor.from_pretrained(ZERO_SHOT_ID, **kwargs)
    model = AutoModelForAudioClassification.from_pretrained(ZERO_SHOT_ID, **kwargs).eval()
    return extractor, model, _synthetic_index(model)


def _synthetic_index(model) -> int:
    """Index of the 'synthetic' class: a label naming TTS/fake/spoof if present, else index 1."""
    labels = {int(k): str(v).lower() for k, v in (getattr(model.config, "id2label", None) or {}).items()}
    for index, name in labels.items():
        if any(word in name for word in ("tts", "fake", "spoof", "synthetic", "generated")):
            return index
    return SYNTHETIC_INDEX


MIN_TAIL = SR // 2          # a final partial window is scored if it holds at least 0.5 s
SILENCE_DBFS = -40.0        # windows quieter than this (after -20 dBFS normalization) are silence


def window_spans(n: int) -> list[tuple[int, int]]:
    """2-second windows covering the whole clip, including a final partial window of >=0.5 s."""
    spans, start = [], 0
    while start < n:
        end = min(n, start + WINDOW)
        if end - start >= MIN_TAIL or not spans:
            spans.append((start, end))
        start += WINDOW
    return spans or [(0, n)]


def zero_shot_windows(y16: np.ndarray) -> np.ndarray:
    """P(synthetic) for each window from window_spans() of a prepared 16 kHz clip."""
    import torch

    extractor, model, synthetic = load_zero_shot()
    probs = []
    with torch.inference_mode():
        for start, end in window_spans(len(y16)):
            window = y16[start:end]
            window = np.pad(window, (0, max(0, WINDOW - len(window))))
            logits = model(**extractor(window, sampling_rate=SR, return_tensors="pt")).logits
            probs.append(float(logits.softmax(-1)[0, synthetic]))
    return np.asarray(probs, dtype=np.float32)


def zero_shot_summary(y16: np.ndarray, probs: np.ndarray | None = None) -> dict:
    """Window scores with silence marked, and a summary over SPEECH windows only.
    Silent windows say nothing about the voice; averaging them in dilutes the score."""
    probs = zero_shot_windows(y16) if probs is None else np.asarray(probs, dtype=np.float32)
    spans = window_spans(len(y16))
    if len(spans) != len(probs):  # defensive: never mis-pair scores and times
        spans = [(i * WINDOW, min(len(y16), (i + 1) * WINDOW)) for i in range(len(probs))]
    speech = []
    for start, end in spans:
        segment = y16[start:end]
        level = 20 * np.log10(np.sqrt(np.mean(np.square(segment, dtype=np.float64))) + 1e-12) if len(segment) else -120
        speech.append(level > SILENCE_DBFS)
    speech = np.asarray(speech, dtype=bool)
    used = probs[speech] if speech.any() else probs
    return {
        "zero_shot_mean": float(used.mean()),
        "zero_shot_max": float(used.max()),
        "zero_shot_share_high": float(np.mean(used >= 0.5)),
        "speech_windows": int(speech.sum()),
        "windows": [{"start_s": s / SR, "end_s": e / SR, "p": float(p), "speech": bool(v)}
                    for (s, e), p, v in zip(spans, probs, speech)],
    }


# ---------------------------------------------------------------- own detector


@st.cache_resource(show_spinner="Loading the WavLM embedding model…")
def load_embedder():
    from transformers import AutoFeatureExtractor, AutoModel

    kwargs = {"revision": EMBED_REVISION} if EMBED_REVISION else {}
    return AutoFeatureExtractor.from_pretrained(EMBED_ID, **kwargs), AutoModel.from_pretrained(EMBED_ID, **kwargs).eval()


def embed(y16: np.ndarray, chunk_seconds: int = 30) -> np.ndarray:
    """Mean-pooled WavLM embedding over 30-second chunks (whole clip, no truncation)."""
    import torch

    extractor, model = load_embedder()
    vectors, weights = [], []
    with torch.inference_mode():
        for start in range(0, max(len(y16), 1), chunk_seconds * SR):
            chunk = y16[start:start + chunk_seconds * SR]
            if len(chunk) < 400:
                continue
            inputs = extractor(chunk, sampling_rate=SR, return_tensors="pt")
            vectors.append(model(**inputs).last_hidden_state[0].mean(0).cpu().numpy())
            weights.append(len(chunk))
    if not vectors:
        raise ValueError("Clip is too short to embed.")
    return np.average(np.stack(vectors), axis=0, weights=weights).astype(np.float32)


def eer(labels, scores) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    index = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    return float((fpr[index] + 1 - tpr[index]) / 2)


def _classifier():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def _score_split(X, y, train, test) -> dict | None:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    if len(set(y[train])) < 2 or len(set(y[test])) < 2:
        return None
    probs = _classifier().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return {"auc": float(roc_auc_score(y[test], probs)), "eer": eer(y[test], probs),
            "brier": float(brier_score_loss(y[test], probs)), "test_clips": int(len(test)),
            "test_real": int((y[test] == 0).sum()), "test_synthetic": int((y[test] == 1).sum())}


def evaluate(X, y, groups, generators, n_splits: int = 5) -> pd.DataFrame:
    """y: 1 synthetic / 0 real. groups: speaker or source recording. generators: 'real', 'bulbul-v3', …"""
    from sklearn.model_selection import GroupKFold

    X, y, groups, generators = (np.asarray(a) for a in (X, y, groups, generators))
    folds = min(n_splits, len(set(groups)))
    if folds < 2:
        raise ValueError("Need at least two speaker/recording groups to evaluate.")
    splits = list(GroupKFold(n_splits=folds).split(X, y, groups))
    rows = []
    for fold, (train, test) in enumerate(splits, 1):
        scored = _score_split(X, y, train, test)
        if scored:
            rows.append({"protocol": "speaker/content-disjoint", "held_out": f"fold {fold}", **scored})
    for held in sorted(set(generators) - {"real"}):
        for fold, (train, test) in enumerate(splits, 1):
            train_rows = train[generators[train] != held]
            test_rows = test[np.isin(generators[test], [held, "real"])]
            scored = _score_split(X, y, train_rows, test_rows)
            if scored:
                rows.append({"protocol": "unseen generator", "held_out": held, "fold": fold, **scored})
    return pd.DataFrame(rows)


def fit_final(X, y):
    return _classifier().fit(np.asarray(X), np.asarray(y))


def detector_bytes(bundle: dict) -> bytes:
    import joblib

    buffer = io.BytesIO()
    joblib.dump(bundle, buffer)
    return buffer.getvalue()


def detector_from_bytes(raw: bytes) -> dict:
    import joblib

    bundle = joblib.load(io.BytesIO(raw))
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("This file is not a detector saved by this app.")
    return bundle


# ---------------------------------------------------------------- scoring used by Research mode


def score_clips(clips, detector: dict | None = None) -> list[dict]:
    """clips: [(name, provenance, audio, sr)]. Returns one row per clip."""
    rows = []
    for name, provenance, audio, sr in clips:
        y16 = prepare(audio, sr)
        row = {"clip": name, "provenance": provenance, "duration_s": len(y16) / SR}
        try:
            row |= zero_shot_summary(y16)
        except Exception as exc:
            row["zero_shot_error"] = f"{type(exc).__name__}: {exc}"
        if detector is not None:
            try:
                prepared = mp3_roundtrip(y16) if detector.get("roundtrip") else y16
                row["own_detector"] = float(detector["model"].predict_proba(embed(prepared)[None, :])[0, 1])
            except Exception as exc:
                row["own_detector_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def render_scores(rows) -> None:
    if not rows:
        st.info("The synthetic-speech check did not produce results; see the run log.")
        return
    import altair as alt

    table = pd.DataFrame([{
        "Clip": r["clip"], "Provenance": r["provenance"], "Duration (s)": r["duration_s"],
        "Speech windows": r.get("speech_windows"),
        "Zero-shot P(synthetic), speech mean": r.get("zero_shot_mean"),
        "Windows ≥ 0.5 (%)": (r["zero_shot_share_high"] * 100) if r.get("zero_shot_share_high") is not None else None,
        "Zero-shot, max window": r.get("zero_shot_max"),
        "Own detector P(synthetic)": r.get("own_detector"),
    } for r in rows])
    if table["Own detector P(synthetic)"].isna().all():
        table = table.drop(columns=["Own detector P(synthetic)"])
    config = {c: st.column_config.NumberColumn(format="%.3f") for c in table.columns
              if c not in ("Clip", "Provenance", "Speech windows", "Windows ≥ 0.5 (%)")}
    config["Clip"] = st.column_config.TextColumn("Clip", width="medium")
    config["Windows ≥ 0.5 (%)"] = st.column_config.NumberColumn(format="%.0f")
    st.dataframe(table, hide_index=True, width="stretch", column_config=config)
    for r in rows:
        for key in ("zero_shot_error", "own_detector_error"):
            if r.get(key):
                st.warning(f"{r['clip']}: {r[key]}")
    windows = pd.DataFrame([
        {"clip": r["clip"], "start (s)": w["start_s"], "end (s)": w["end_s"], "P(synthetic)": w["p"],
         "window": "speech" if w["speech"] else "silence (not in mean)"}
        for r in rows for w in r.get("windows", []) if isinstance(w, dict)
    ])
    rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
    for clip, clip_windows in windows.groupby("clip", sort=False) if not windows.empty else []:
        bars = alt.Chart(clip_windows).mark_bar(color="#2a78d6", stroke="white", strokeWidth=1).encode(
            x=alt.X("start (s):Q", title="time (s)"), x2="end (s):Q",
            y=alt.Y("P(synthetic):Q", scale=alt.Scale(domain=[0, 1])),
            opacity=alt.Opacity("window:N", scale=alt.Scale(domain=["speech", "silence (not in mean)"], range=[0.85, 0.2]),
                                legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[alt.Tooltip("start (s):Q", format=".1f"), alt.Tooltip("end (s):Q", format=".1f"),
                     alt.Tooltip("P(synthetic):Q", format=".3f"), "window"],
        )
        st.altair_chart((bars + rule).properties(height=200, title=f"{clip} — zero-shot score per 2-second window"),
                        width="stretch")
    st.caption(
        "Each bar is one 2-second window (the last may be shorter); faded bars are silence and are left out of the "
        "mean. Near 1 = 'looks synthetic' to this detector. Generated rows are known synthetic, so they show whether "
        "the detector catches this app's engines. Uploaded or recorded clips are provenance-unknown: the zero-shot "
        "model has no independent test on outside data, and microphones, MP3 compression and code-mixed speech can "
        "raise its score. A score is not proof that a recording is genuine or fake; train your own detector and "
        "compare."
    )


# ---------------------------------------------------------------- dataset held in the session


def _dataset() -> list[dict]:
    return st.session_state.setdefault("deepfake_dataset", [])


def add_clip(name, label, generator, group, audio, sr, roundtrip: bool, zero_shot: bool) -> str:
    """Embed one clip and store it (vectors only, not audio). Returns 'added' or 'duplicate'."""
    y16 = prepare(audio, sr)
    clip_id = hashlib.sha256(y16.tobytes() + f"{roundtrip}".encode()).hexdigest()[:16]
    dataset = _dataset()
    if any(item["id"] == clip_id for item in dataset):
        return "duplicate"
    prepared = mp3_roundtrip(y16) if roundtrip else y16
    item = {"id": clip_id, "name": name, "label": int(label), "generator": generator, "group": str(group),
            "embedding": embed(prepared), "roundtrip": bool(roundtrip), "zero_shot": float("nan")}
    if zero_shot:
        item["zero_shot"] = zero_shot_summary(y16)["zero_shot_mean"]
    dataset.append(item)
    return "added"


def dataset_frame() -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in item.items() if k != "embedding"} for item in _dataset()])


def dataset_npz() -> bytes:
    dataset = _dataset()
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        embeddings=np.stack([item["embedding"] for item in dataset]),
        ids=np.asarray([item["id"] for item in dataset]), names=np.asarray([item["name"] for item in dataset]),
        labels=np.asarray([item["label"] for item in dataset]),
        generators=np.asarray([item["generator"] for item in dataset]),
        groups=np.asarray([item["group"] for item in dataset]),
        roundtrip=np.asarray([item["roundtrip"] for item in dataset]),
        zero_shot=np.asarray([item["zero_shot"] for item in dataset], dtype=np.float32),
        embed_id=np.asarray(EMBED_ID),
    )
    return buffer.getvalue()


def load_dataset_npz(raw: bytes) -> int:
    data = np.load(io.BytesIO(raw), allow_pickle=False)
    if str(data["embed_id"]) != EMBED_ID:
        raise ValueError(f"Dataset was embedded with {data['embed_id']}, but this app uses {EMBED_ID}.")
    dataset, known, added = _dataset(), {item["id"] for item in _dataset()}, 0
    for index, clip_id in enumerate(data["ids"]):
        if str(clip_id) in known:
            continue
        dataset.append({"id": str(clip_id), "name": str(data["names"][index]), "label": int(data["labels"][index]),
                        "generator": str(data["generators"][index]), "group": str(data["groups"][index]),
                        "embedding": data["embeddings"][index].astype(np.float32),
                        "roundtrip": bool(data["roundtrip"][index]), "zero_shot": float(data["zero_shot"][index])})
        added += 1
    return added


# ---------------------------------------------------------------- Streamlit mode


def render_synthetic_mode() -> None:
    st.subheader("Synthetic speech (deepfake) detection")
    st.caption(
        "Build a labelled set from your real recordings and this app's TTS outputs, train a detector, and "
        "test it on speakers and TTS engines it has not seen. Results are research evidence, not proof."
    )
    tabs = st.tabs(["Check a file", "Build dataset", "Train & evaluate"])
    with tabs[0]:
        _render_check_tab()
    with tabs[1]:
        _render_dataset_tab()
    with tabs[2]:
        _render_train_tab()


def _render_check_tab() -> None:
    detector = st.session_state.get("deepfake_detector")
    saved = st.file_uploader("Load a saved detector (.joblib) — only files you created with this app",
                             type=["joblib"], key="deepfake_detector_upload")
    if saved is not None:
        try:
            detector = detector_from_bytes(saved.getvalue())
            st.session_state["deepfake_detector"] = detector
            st.success(f"Detector loaded: trained on {detector.get('clips', '?')} clips.")
        except Exception as exc:
            st.error(f"Could not load detector: {type(exc).__name__}: {exc}")
    if detector is None:
        st.info("No own detector yet: only the zero-shot baseline will score. Train one in the last tab.")
    left, right = st.columns(2)
    recording = left.audio_input("Record speech", key="deepfake_recording")
    upload = right.file_uploader("Or upload audio", type=AUDIO_TYPES, key="deepfake_check_upload")
    source = recording or upload
    if st.button("Score this clip", type="primary", disabled=source is None, key="deepfake_check_run"):
        try:
            with st.spinner("Scoring…"):
                audio, sr = decode_upload(source.getvalue(), Path(getattr(source, "name", "recording.wav")).suffix)
                st.session_state["deepfake_check_rows"] = score_clips(
                    [(getattr(source, "name", "recording"), "unknown", audio, sr)], detector)
        except Exception as exc:
            st.error(f"Scoring failed: {type(exc).__name__}: {exc}")
    if st.session_state.get("deepfake_check_rows"):
        render_scores(st.session_state["deepfake_check_rows"])


def _render_dataset_tab() -> None:
    columns = st.columns(2)
    roundtrip = columns[0].toggle("MP3 codec control", value=ffmpeg_available(), disabled=not ffmpeg_available(),
                                  key="deepfake_roundtrip",
                                  help="Pass every clip through the same 64 kbit/s MP3 encode. Needs ffmpeg.")
    zero_shot = columns[1].toggle("Also store zero-shot scores", value=True, key="deepfake_zero_shot",
                                  help="Lets Train & evaluate compare your detector with the zero-shot baseline.")
    if not ffmpeg_available():
        st.warning("ffmpeg was not found, so the MP3 codec control is off. Install ffmpeg for thesis runs.")
    dataset = _dataset()
    if dataset and any(item["roundtrip"] != roundtrip for item in dataset):
        st.warning("The dataset mixes clips with and without the MP3 codec control. Keep one setting for a fair test.")

    st.markdown("**1 · Add the last Batch Research run**")
    batch = st.session_state.get("batch_results")
    if not batch or not batch[0]:
        st.caption("Run Batch Research with one or more TTS models first. Each input recording is added as real, "
                   "each TTS output as synthetic, grouped by the recording's speaker ID.")
    else:
        results = batch[0]
        n_tts = sum(r["status"] == "ok" for result in results for r in result["tts"])
        st.caption(f"Last batch: {len(results)} recordings and {n_tts} generated clips.")
        if st.button("Add batch clips to the dataset", key="deepfake_add_batch"):
            counts = {"added": 0, "duplicate": 0, "failed": 0}
            progress = st.progress(0.0, text="Embedding clips")
            total = len(results) + n_tts
            done = 0
            for result in results:
                group = result.get("speaker") or Path(result["name"]).stem
                clips = [(result["name"], 0, "real", result["pre"]["normalized"], result["sr"])]
                clips += [(f"{result['name']} → {r['label']}", 1, r["key"], r["audio"], r["sr"])
                          for r in result["tts"] if r["status"] == "ok"]
                for name, label, generator, audio, sr in clips:
                    try:
                        counts[add_clip(name, label, generator, group, audio, sr, roundtrip, zero_shot)] += 1
                    except Exception as exc:
                        counts["failed"] += 1
                        st.warning(f"{name}: {type(exc).__name__}: {exc}")
                    done += 1
                    progress.progress(done / max(total, 1), text=f"Embedded {done}/{total}")
            st.success(f"Added {counts['added']}, skipped {counts['duplicate']} duplicates, {counts['failed']} failed.")

    st.markdown("**2 · Add labelled files** (more real speakers, or other generators such as IndicSynth Hindi)")
    uploads = st.file_uploader("Labelled audio", type=AUDIO_TYPES, accept_multiple_files=True, key="deepfake_uploads")
    if uploads:
        labels = st.data_editor(
            pd.DataFrame([{"file": u.name, "label": "real", "generator": "real", "group": Path(u.name).stem}
                          for u in uploads]),
            hide_index=True, width="stretch", num_rows="fixed", disabled=["file"], key="deepfake_upload_labels",
            column_config={"label": st.column_config.SelectboxColumn("label", options=["real", "synthetic"], required=True)},
        )
        st.caption("For synthetic files, set 'generator' to the system that made them (for example indicsynth-vits). "
                   "'group' is the speaker or source recording; clips sharing a group never sit on both sides of a split.")
        if st.button("Add labelled files", key="deepfake_add_uploads"):
            added = 0
            for upload, (_, meta) in zip(uploads, labels.iterrows()):
                is_fake = meta["label"] == "synthetic"
                generator = str(meta["generator"]).strip() or ("unknown-generator" if is_fake else "real")
                if is_fake and generator == "real":
                    generator = "unknown-generator"
                if not is_fake:
                    generator = "real"
                try:
                    audio, sr = decode_upload(upload.getvalue(), Path(upload.name).suffix)
                    added += add_clip(upload.name, int(is_fake), generator, meta["group"], audio, sr,
                                      roundtrip, zero_shot) == "added"
                except Exception as exc:
                    st.warning(f"{upload.name}: {type(exc).__name__}: {exc}")
            st.success(f"Added {added} file(s).")

    st.markdown("**3 · Save or restore the dataset** (embeddings only, no audio)")
    restore = st.file_uploader("Restore a saved dataset (.npz)", type=["npz"], key="deepfake_dataset_upload")
    if restore is not None and st.button("Restore", key="deepfake_restore"):
        try:
            st.success(f"Restored {load_dataset_npz(restore.getvalue())} clip(s).")
        except Exception as exc:
            st.error(f"Could not restore: {type(exc).__name__}: {exc}")

    frame = dataset_frame()
    if frame.empty:
        st.info("The dataset is empty.")
        return
    summary = frame.groupby(["generator", "label"]).agg(clips=("id", "count"), groups=("group", "nunique")).reset_index()
    st.dataframe(summary, hide_index=True, width="stretch")
    real_groups = frame[frame.label == 0].group.nunique()
    st.caption(f"{len(frame)} clips · {real_groups} real speaker/recording groups.")
    left, right = st.columns(2)
    left.download_button("Download dataset (.npz)", dataset_npz(), "synthetic_speech_dataset.npz", key="deepfake_dataset_download")
    if right.button("Clear dataset", key="deepfake_clear"):
        st.session_state["deepfake_dataset"] = []
        st.rerun()


def _render_train_tab() -> None:
    frame = dataset_frame()
    if frame.empty or frame.label.nunique() < 2:
        st.info("Add both real and synthetic clips in Build dataset first.")
        return
    real_groups = frame[frame.label == 0].group.nunique()
    if real_groups < MIN_REAL_GROUPS_FOR_CLAIMS:
        st.warning(f"Only {real_groups} real speaker/recording groups (aim for {MIN_REAL_GROUPS_FOR_CLAIMS}+). "
                   "Treat the numbers as indicative, not as a detection rate.")
    folds = st.slider("Cross-validation folds", 2, 10, 5, key="deepfake_folds")
    if st.button("Train and evaluate", type="primary", key="deepfake_train"):
        dataset = _dataset()
        X = np.stack([item["embedding"] for item in dataset])
        y = np.asarray([item["label"] for item in dataset])
        groups = np.asarray([item["group"] for item in dataset])
        generators = np.asarray([item["generator"] for item in dataset])
        try:
            with st.spinner("Cross-validating…"):
                results = evaluate(X, y, groups, generators, folds)
                model = fit_final(X, y)
            bundle = {"model": model, "embed_id": EMBED_ID, "roundtrip": bool(dataset[0]["roundtrip"]),
                      "clips": len(dataset), "generators": sorted(set(generators) - {"real"}),
                      "real_groups": int(real_groups)}
            st.session_state["deepfake_detector"] = bundle
            st.session_state["deepfake_results"] = results
            st.session_state["deepfake_baseline"] = _zero_shot_baseline(frame)
        except Exception as exc:
            st.error(f"Training failed: {type(exc).__name__}: {exc}")

    results = st.session_state.get("deepfake_results")
    if results is None:
        return
    if results.empty:
        st.warning("No fold had both real and synthetic clips in its test set. Add more groups.")
        return
    summary = results.groupby(["protocol", "held_out"]).agg(
        auc=("auc", "mean"), eer=("eer", "mean"), brier=("brier", "mean"), folds=("auc", "count"),
        test_clips=("test_clips", "sum"),
    ).reset_index()
    disjoint = results[results.protocol == "speaker/content-disjoint"]
    st.markdown("**Own detector**")
    if not disjoint.empty:
        st.metric("Speaker/content-disjoint EER (mean of folds)", f"{disjoint.eer.mean() * 100:.1f}%",
                  help="Equal error rate: where false accepts equal false rejects. Lower is better; 50% is chance.")
    unseen = summary[summary.protocol == "unseen generator"]
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.3f") for c in ("auc", "eer", "brier")
    })
    if not unseen.empty:
        st.bar_chart(unseen, x="held_out", y="eer")
        st.caption("EER on each TTS engine when it was left out of training. A high bar means the detector "
                   "does not generalize to that engine; report this, not only the pooled number.")
    baseline = st.session_state.get("deepfake_baseline")
    if baseline:
        st.markdown("**Zero-shot baseline** (no training, whole dataset)")
        st.dataframe(pd.DataFrame([baseline]), hide_index=True, width="stretch")
    with st.expander("All folds"):
        st.dataframe(results, hide_index=True, width="stretch")
    st.download_button("Download evaluation CSV", results.to_csv(index=False).encode(), "deepfake_evaluation.csv",
                       key="deepfake_results_download")
    bundle = st.session_state.get("deepfake_detector")
    if bundle:
        st.download_button("Download trained detector (.joblib)", detector_bytes(bundle), "detector.joblib",
                           key="deepfake_detector_download")
        st.caption("The final detector is trained on all clips. It is used by Check a file and by the "
                   "Synthetic-speech check in Research mode during this session.")


def _zero_shot_baseline(frame: pd.DataFrame) -> dict | None:
    from sklearn.metrics import roc_auc_score

    scored = frame[np.isfinite(frame.zero_shot.astype(float))]
    if scored.empty or scored.label.nunique() < 2:
        return None
    return {"clips": int(len(scored)), "auc": round(float(roc_auc_score(scored.label, scored.zero_shot)), 3),
            "eer": round(eer(scored.label, scored.zero_shot), 3), "model": ZERO_SHOT_ID}
