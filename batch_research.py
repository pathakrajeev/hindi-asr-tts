"""Multi-file research extension for the existing Hindi ASR/TTS Streamlit app.

MFA is an optional external command, requiring compatible per-language acoustic
models and pronunciation dictionaries. Wav2Vec2 embeddings are optional.
"""
import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from speech_core import SAMPLE_RATE, decode_audio, normalize_reference_text
from research import (ASR_ENGINES, TTS_ENGINES, run_research, render_asr,
                      render_tts, render_prosody, render_preprocessing)

AUDIO_EXTENSIONS = ["wav", "mp3", "m4a", "mp4", "flac", "ogg", "aac"]


def acoustic_embedding(audio, sr, mfcc_dim=40):
    """Fixed-length acoustic descriptor, NOT a learned speaker embedding."""
    y = librosa.resample(audio, orig_sr=sr, target_sr=16000) if sr != 16000 else audio
    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=mfcc_dim)
    delta = librosa.feature.delta(mfcc) if mfcc.shape[1] >= 9 else np.zeros_like(mfcc)
    features = np.concatenate([mfcc, delta], axis=0)
    return np.r_[features.mean(axis=1), features.std(axis=1)].astype(np.float32)


@st.cache_resource(show_spinner=False)
def wav2vec_bundle(model_id):
    from transformers import AutoFeatureExtractor, AutoModel
    import torch
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()
    return extractor, model


def wav2vec_embedding(audio, sr, model_id):
    import torch
    y = librosa.resample(audio, orig_sr=sr, target_sr=16000) if sr != 16000 else audio
    # Bound inference cost; longer recordings use the first 30 seconds.
    y = y[:30 * 16000]
    extractor, model = wav2vec_bundle(model_id)
    inputs = extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.inference_mode():
        states = model(**inputs).last_hidden_state[0].mean(dim=0)
    return states.cpu().numpy().astype(np.float32)


def similarity_rows(items):
    rows = []
    for i, left in enumerate(items):
        for j, right in enumerate(items):
            a, b = left["embedding"], right["embedding"]
            score = float(np.dot(a, b) / (max(np.linalg.norm(a), 1e-9) * max(np.linalg.norm(b), 1e-9)))
            rows.append({"File A": left["file"], "File B": right["file"], "cosine similarity": score})
    return pd.DataFrame(rows)


def run_mfa(audio, sr, transcript, dictionary, acoustic, command="mfa"):
    """Align reference words using an installed MFA CLI. Returns TextGrid intervals."""
    import textgrid
    executable = shutil.which(command)
    if not executable:
        raise RuntimeError(
            "MFA command not found on PATH. Rebuild the Space using this project's Dockerfile "
            "to install Montreal Forced Aligner."
        )
    if not dictionary or not acoustic:
        raise ValueError("An MFA pronunciation dictionary and compatible acoustic model are required.")
    with tempfile.TemporaryDirectory(prefix="speech_mfa_") as folder:
        root = Path(folder)
        corpus, output = root / "corpus", root / "aligned"
        corpus.mkdir()
        output.mkdir()
        sf.write(corpus / "sample.wav", audio, sr, subtype="PCM_16")
        (corpus / "sample.lab").write_text(transcript.strip(), encoding="utf-8")
        cmd = [executable, "align", str(corpus), dictionary, acoustic, str(output), "--clean"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode:
            raise RuntimeError((proc.stderr or proc.stdout)[-900:])
        path = output / "sample.TextGrid"
        if not path.exists():
            matches = list(output.rglob("*.TextGrid"))
            if not matches:
                raise RuntimeError("MFA finished without producing a TextGrid.")
            path = matches[0]
        grid = textgrid.TextGrid.fromFile(str(path))
        tier = next((t for t in grid.tiers if t.name.lower() in ("words", "word")), None)
        if tier is None:
            raise RuntimeError("No words tier in MFA output.")
        return [{"word": x.mark, "start_s": float(x.minTime), "end_s": float(x.maxTime),
                 "duration_s": float(x.maxTime - x.minTime)} for x in tier if x.mark.strip()]


def render_batch_tab():
    st.subheader("Batch research · multi-file analysis")
    st.caption("Upload audio and matching transcript files to prefill references automatically. Compare ASR/TTS, acoustic embeddings and optional IndicMFA alignment.")
    uploads = st.file_uploader("Audio recordings", type=AUDIO_EXTENSIONS, accept_multiple_files=True, key="batch_audio")
    reference_files = st.file_uploader(
        "Reference transcripts (.txt, UTF-8; automatically matched by filename)",
        type=["txt"],
        accept_multiple_files=True,
        help="For example, upload recording.wav and recording.txt. The text file contains only the correct transcript.",
        key="batch_txt",
    )
    if not uploads:
        st.session_state.pop("batch_results", None)
        st.info("Upload WAV/MP3/FLAC/etc. files. Each file is limited to 120 seconds and 50 MB.")
        return
    try:
        ref_map = {}
        if reference_files:
            for reference_file in reference_files:
                key = Path(reference_file.name).stem.casefold()
                if key in ref_map:
                    st.error(f"More than one reference file matches the filename stem '{Path(reference_file.name).stem}'.")
                    return
                try:
                    text = reference_file.getvalue().decode("utf-8-sig").strip()
                except UnicodeDecodeError:
                    st.error(f"Reference file '{reference_file.name}' must be UTF-8 text.")
                    return
                ref_map[key] = (text, "hi", "TXT")
        metadata_rows = []
        for upload in uploads:
            audio_stem = Path(upload.name).stem.casefold()
            reference = ref_map.get(upload.name.casefold(), ref_map.get(audio_stem, ("", "hi", "")))
            metadata_rows.append({
                "filename": upload.name,
                "reference": reference[0],
                "language": reference[1] or "hi",
            })
        metadata = pd.DataFrame(metadata_rows)
        matched_count = sum(bool(normalize_reference_text(row["reference"])) for row in metadata_rows)
        if matched_count:
            st.success(f"Automatically matched references for {matched_count} of {len(uploads)} audio files.")
        else:
            st.info("No references matched yet. Upload same-stem .txt files (for example, clip1.wav + clip1.txt), or edit the table below.")
        st.caption("Language defaults to Hindi (hi); change it to Sanskrit (sa) in the table when needed.")
        edited = st.data_editor(metadata, hide_index=True, width="stretch", num_rows="fixed", key="batch_metadata")
        asr = st.multiselect("Batch ASR models", list(ASR_ENGINES), default=["koyal"],
                             format_func=lambda x: ASR_ENGINES[x]["label"], key="batch_asr")
        tts = st.multiselect("Batch TTS models", list(TTS_ENGINES), default=[],
                             format_func=lambda x: TTS_ENGINES[x]["label"], key="batch_tts")
        denoise = st.toggle("Adaptive denoising", value=True, key="batch_denoise")
        embed_mode = st.selectbox("Audio embeddings", ["MFCC + delta statistics", "Wav2Vec2 learned features", "Off"], key="batch_embed")
        model_id = "facebook/wav2vec2-base-960h"
        if embed_mode == "Wav2Vec2 learned features":
            model_id = st.text_input("Hugging Face embedding model ID", value=model_id)
            st.caption("Mean-pooled Wav2Vec2 representations are general speech embeddings, not verified speaker identity scores.")
        enable_mfa = st.toggle("IndicMFA forced word alignment", value=False, key="batch_mfa")
        dictionaries, acoustics = {}, {}
        if enable_mfa:
            st.info("MFA is installed by this project's Dockerfile for Hugging Face Spaces. Project-local dictionaries and acoustic model archives are prefilled when present; paths can be changed below.")
            project_dir = Path(__file__).resolve().parent
            mfa_assets = {
                "hi": ("Hindi_Dict_g2g.txt", "Hindi_All_Acoustic.zip"),
                "sa": ("Sanskrit_Dict.txt", "Sanskrit_Acoustic_Model.zip"),
            }
            for lang in ("hi", "sa"):
                a, b = st.columns(2)
                dictionary_file, acoustic_file = mfa_assets[lang]
                dictionary_default = os.getenv(
                    f"INDICMFA_{lang.upper()}_DICTIONARY",
                    str(project_dir / dictionary_file) if (project_dir / dictionary_file).is_file() else "",
                )
                acoustic_default = os.getenv(
                    f"INDICMFA_{lang.upper()}_ACOUSTIC",
                    str(project_dir / acoustic_file) if (project_dir / acoustic_file).is_file() else "",
                )
                dictionaries[lang] = a.text_input(
                    f"{lang} dictionary path/model",
                    value=dictionary_default,
                    key=f"batch_mfa_{lang}_dictionary",
                )
                acoustics[lang] = b.text_input(
                    f"{lang} acoustic model path/name",
                    value=acoustic_default,
                    key=f"batch_mfa_{lang}_acoustic",
                )
        run = st.button("Run batch analysis", type="primary", disabled=not bool(asr) or not bool(uploads))
        if run:
            st.session_state.pop("batch_results", None)
            results, embeddings, alignment_rows, errors = [], [], [], []
            progress = st.progress(0, text="Starting batch")
            for idx, upload in enumerate(uploads):
                name = upload.name
                meta = edited.iloc[idx]
                text = str(meta["reference"]).strip()
                language = str(meta["language"]).strip().lower()
                if not normalize_reference_text(text):
                    errors.append({"file": name, "error": "Missing golden-reference transcript"})
                    progress.progress((idx + 1) / len(uploads))
                    continue
                try:
                    audio, sr, duration = decode_audio(upload.getvalue(), Path(name).suffix, 120)
                    options = {"asr": asr, "tts": tts, "tts_source": "reference", "judge": None,
                               "sarvam_mode": "codemix", "denoise": denoise, "utmos": False, "release": True}
                    with st.status(f"{name}: running models", expanded=False) as status:
                        result = run_research(audio, sr, text, options, status)
                    result["name"] = name
                    result["language"] = language
                    results.append(result)
                    if embed_mode != "Off":
                        vector = (acoustic_embedding(audio, sr) if embed_mode.startswith("MFCC")
                                  else wav2vec_embedding(audio, sr, model_id))
                        embeddings.append({"file": name, "embedding": vector})
                    if enable_mfa:
                        if language not in ("hi", "sa"):
                            alignment_rows.append({"file": name, "status": "unsupported language", "language": language})
                        else:
                            try:
                                intervals = run_mfa(audio, sr, text, dictionaries[language], acoustics[language])
                                alignment_rows.extend({"file": name, "language": language, "status": "ok", **word} for word in intervals)
                            except Exception as exc:
                                alignment_rows.append({"file": name, "language": language, "status": "failed", "error": str(exc)})
                except Exception as exc:
                    errors.append({"file": name, "error": f"{type(exc).__name__}: {exc}"})
                progress.progress((idx + 1) / len(uploads), text=f"Processed {idx+1}/{len(uploads)}")
            st.session_state["batch_results"] = (results, embeddings, alignment_rows, errors)
        stored = st.session_state.get("batch_results")
        if stored is None:
            return
        results, embeddings, alignment_rows, errors = stored
        if errors:
            st.warning("Some files could not be processed.")
            st.dataframe(pd.DataFrame(errors), hide_index=True)
        if not results:
            return
        aggregate = []
        for result in results:
            for row in result["asr"]:
                aggregate.append({"file": result["name"], "language": result["language"],
                                  "model": row["label"], "status": row["status"], "WER (%)": row.get("wer"),
                                  "CER (%)": row.get("cer"), "Latency (s)": row["latency"], "RTF": row["rtf"]})
        scores = pd.DataFrame(aggregate)
        st.subheader("Across-file ASR benchmark")
        st.dataframe(scores, hide_index=True, width="stretch")
        st.download_button("Download ASR results CSV", scores.to_csv(index=False).encode(), "batch_asr.csv", "text/csv")
        good = scores[scores.status == "ok"]
        if not good.empty:
            st.bar_chart(good, x="model", y="WER (%)", color="file", stack=False)
            st.caption("For mixed Hindi / code-mixed / Sanskrit experiments, interpret WER by language and scoring protocol; pooled WER may not be meaningful.")
        if embeddings:
            st.subheader("Speech embedding comparison")
            vectors = similarity_rows(embeddings)
            pivot = vectors.pivot(index="File A", columns="File B", values="cosine similarity")
            st.dataframe(pivot.style.format("{:.3f}"), width="stretch")
            st.download_button("Download embeddings (NPZ)", _embedding_npz(embeddings), "speech_embeddings.npz")
            st.caption("Cosine similarity describes feature similarity; it is not a validated speaker-verification or deepfake-detection score.")
        if alignment_rows:
            st.subheader("IndicMFA word boundaries")
            alignment = pd.DataFrame(alignment_rows)
            st.dataframe(alignment, hide_index=True, width="stretch")
            st.download_button("Download alignment CSV", alignment.to_csv(index=False).encode(), "mfa_word_alignment.csv", "text/csv")
        st.subheader("Individual recording results")
        selected = st.selectbox("Inspect file", [r["name"] for r in results])
        result = next(r for r in results if r["name"] == selected)
        tabs = st.tabs(["Preprocessing", "ASR", "TTS", "Prosody", "Execution log"])
        with tabs[0]: render_preprocessing(result)
        with tabs[1]: render_asr(result)
        with tabs[2]: render_tts(result)
        with tabs[3]: render_prosody(result)
        with tabs[4]: st.dataframe(pd.DataFrame(result["log"]), hide_index=True)
    except Exception as exc:
        st.error(f"Batch setup error: {type(exc).__name__}: {exc}")


def _embedding_npz(items):
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"embedding_{i}": r["embedding"] for i, r in enumerate(items)},
                        filenames=np.asarray([r["file"] for r in items]))
    return buf.getvalue()
