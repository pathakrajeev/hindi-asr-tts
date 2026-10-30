"""Multi-file research extension for the existing Hindi ASR/TTS Streamlit app.

MFA is an optional external command, requiring compatible per-language acoustic
models and pronunciation dictionaries. Wav2Vec2 embeddings are optional.
"""
import io
import json
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

import stats
from speech_core import (LANGUAGES, PRIMARY_LABEL, PRIMARY_METRIC, SAMPLE_RATE, decode_audio, guess_language,
                         normalize_reference_text)
from research import (ASR_ENGINES, FEMALE_VOICE_F0_THRESHOLD_HZ, TTS_ENGINES, _settings_digest,
                      engine_support, _inference_device, release, render_result_tabs, run_research)

AUDIO_EXTENSIONS = ["wav", "mp3", "m4a", "mp4", "flac", "ogg", "aac"]
PROVENANCE = ["unknown", "human_from_scratch", "machine_draft_corrected", "published_text"]
# Pooled corpus metric per variety: (errors column, denominator column).
CORPUS_COLUMNS = {
    "hi": ("errors_norm", "reference_words_norm"),
    "hi-en": ("errors_script_tolerant", "reference_words_norm"),
    "sa": ("char_errors_norm", "reference_chars_norm"),
}


def acoustic_embedding(audio, sr, mfcc_dim=40):
    """Fixed-length acoustic descriptor, NOT a learned speaker embedding."""
    y = librosa.resample(audio, orig_sr=sr, target_sr=16000) if sr != 16000 else audio
    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=mfcc_dim)
    delta = librosa.feature.delta(mfcc) if mfcc.shape[1] >= 9 else np.zeros_like(mfcc)
    features = np.concatenate([mfcc, delta], axis=0)
    return np.r_[features.mean(axis=1), features.std(axis=1)].astype(np.float32)


@st.cache_resource(show_spinner=False)
def wav2vec_bundle(model_id, device="cpu"):
    from transformers import AutoFeatureExtractor, AutoModel
    import torch
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return extractor, model


def wav2vec_embedding(audio, sr, model_id, device="cpu"):
    import torch
    y = librosa.resample(audio, orig_sr=sr, target_sr=16000) if sr != 16000 else audio
    # Whole recording in 30-second chunks (previously only the first 30 s were used),
    # averaged with chunk-length weights so a short tail chunk does not count as much.
    extractor, model = wav2vec_bundle(model_id, device)
    vectors, weights = [], []
    with torch.inference_mode():
        for start in range(0, max(len(y), 1), 30 * 16000):
            chunk = y[start:start + 30 * 16000]
            if len(chunk) < 400:  # under 25 ms: too short for the feature encoder
                continue
            inputs = extractor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            vectors.append(model(**inputs).last_hidden_state[0].mean(dim=0).cpu().numpy())
            weights.append(len(chunk))
    if not vectors:
        raise ValueError("Recording is too short for a Wav2Vec2 embedding.")
    return np.average(np.stack(vectors), axis=0, weights=weights).astype(np.float32)


def similarity_rows(items):
    rows = []
    for i, left in enumerate(items):
        for j, right in enumerate(items):
            a, b = left["embedding"], right["embedding"]
            score = float(np.dot(a, b) / (max(np.linalg.norm(a), 1e-9) * max(np.linalg.norm(b), 1e-9)))
            rows.append({"File A": left["file"], "File B": right["file"], "cosine similarity": score})
    return pd.DataFrame(rows)


def run_mfa(audio, sr, transcript, dictionary, acoustic, command="mfa"):
    """Align reference words using an installed MFA CLI. Returns word intervals.

    The IndicMFA dictionaries list akshara units only, so whole words would align as OOV (spn);
    naturalness.align_one builds a word-to-akshara lexicon from the acoustic model's phone set."""
    import naturalness
    if not dictionary or not acoustic:
        raise ValueError("An MFA pronunciation dictionary and compatible acoustic model are required.")
    words = naturalness.align_one(audio, sr, transcript, acoustic=acoustic, dictionary=dictionary,
                                  command=command, jobs=1, log=lambda *_: None)
    return [{"word": w["word"], "start_s": w["start_s"], "end_s": w["end_s"], "duration_s": w["dur_s"],
             "pause_after_s": w["pause_after_s"], "oov": w["oov"]} for w in words]


def render_batch_tab():
    st.subheader("Batch research · multi-file analysis")
    st.caption(
        "Upload audio and matching transcript files to prefill references automatically. "
        "IndicLID suggests Hindi or Sanskrit from Devanagari transcripts; the existing text rules "
        "detect Hindi-English code-mixing. Its model downloads on first use (~161 MB). "
        "Review or override the suggestion before running models."
    )
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
        st.session_state.pop("batch_digest", None)
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
                ref_map[key] = text
        metadata_rows = []
        with st.spinner("Suggesting transcript languages (first IndicLID use downloads its model)…"):
            for upload in uploads:
                reference = ref_map.get(Path(upload.name).stem.casefold(), "")
                language, evidence = guess_language(reference)
                metadata_rows.append({
                    "filename": upload.name,
                    "reference": reference,
                    "language": language,
                    "language evidence": evidence,
                    "speaker": Path(upload.name).stem,
                    "provenance": "unknown",
                })
        metadata = pd.DataFrame(metadata_rows)
        matched_count = sum(bool(normalize_reference_text(row["reference"])) for row in metadata_rows)
        if matched_count:
            st.success(f"Automatically matched references for {matched_count} of {len(uploads)} audio files.")
        else:
            st.info("No references matched yet. Upload same-stem .txt files (for example, clip1.wav + clip1.txt), or edit the table below.")
        st.caption(
            "**language** is suggested from each reference text (hi = Hindi, hi-en = Hindi-English code-mixed, "
            "sa = Sanskrit); the evidence column says why. Check it and change it if wrong. It decides which "
            "models run and which metric is the headline. **speaker** keeps one voice on one side of the "
            "deepfake train/test split. **provenance** says where the reference came from: human_from_scratch "
            "(you typed it while listening, or wrote the script that was then read aloud, and checked it against "
            "the audio), machine_draft_corrected (an ASR draft you corrected), published_text (from a dataset "
            "or book), unknown. The app cannot tell from the file, so it starts as unknown."
        )
        edited = st.data_editor(
            metadata, hide_index=True, width="stretch", num_rows="fixed", key="batch_metadata",
            disabled=["filename", "language evidence"],
            column_config={
                "language": st.column_config.SelectboxColumn("language", options=list(LANGUAGES), required=True),
                "provenance": st.column_config.SelectboxColumn("provenance", options=PROVENANCE, required=True),
                "speaker": st.column_config.TextColumn("speaker"),
            },
        )
        asr = st.multiselect("Batch ASR models", list(ASR_ENGINES), default=["koyal"],
                             format_func=lambda x: ASR_ENGINES[x]["label"], key="batch_asr")
        tts = st.multiselect("Batch TTS models", list(TTS_ENGINES), default=[],
                             format_func=lambda x: TTS_ENGINES[x]["label"], key="batch_tts")
        columns = st.columns(3)
        judge = columns[0].selectbox(
            "Loopback judge (re-transcribes TTS output)", [None, *asr],
            format_func=lambda k: "None" if k is None else ASR_ENGINES[k]["label"], key="batch_judge",
            help="Gives each TTS output an intelligibility score (loopback WER/CER). Off by default because it re-runs ASR.",
        )
        utmos = columns[1].toggle("UTMOS score", value=False, key="batch_utmos")
        denoise = columns[2].toggle("Adaptive denoising", value=True, key="batch_denoise")
        gpu_available = _inference_device(True).type == "cuda"
        use_gpu = st.toggle(
            "Use GPU for local models",
            value=gpu_available,
            disabled=not gpu_available,
            key="batch_use_gpu",
            help="Run supported local ASR/TTS models and Wav2Vec2 embeddings on CUDA. Audio preprocessing, MFCC embeddings, MFA, and scoring remain on CPU.",
        )
        if gpu_available and use_gpu:
            import torch

            gpu = torch.cuda.get_device_properties(0)
            st.caption(f"CUDA enabled · {torch.cuda.get_device_name(0)} · {gpu.total_memory / 1024**3:.1f} GB VRAM")
        elif gpu_available:
            st.caption("Local models will run on CPU; CUDA is available but disabled.")
        else:
            st.info("CUDA GPU not detected. Local models and Wav2Vec2 embeddings will use CPU.")
        embed_mode = st.selectbox("Audio embeddings", ["MFCC + delta statistics", "Wav2Vec2 learned features", "Off"], key="batch_embed")
        model_id = "facebook/wav2vec2-base-960h"
        if embed_mode == "Wav2Vec2 learned features":
            model_id = st.text_input("Hugging Face embedding model ID", value=model_id, key="batch_embed_model")
            st.caption("Mean-pooled Wav2Vec2 representations are general speech embeddings, not verified speaker identity scores. "
                       "The default model is English-trained; a multilingual model such as facebook/wav2vec2-xls-r-300m suits Hindi better (larger download).")
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
                dictionaries[lang] = a.text_input(f"{lang} dictionary path/model", value=dictionary_default,
                                                  key=f"batch_mfa_{lang}_dictionary")
                acoustics[lang] = b.text_input(f"{lang} acoustic model path/name", value=acoustic_default,
                                               key=f"batch_mfa_{lang}_acoustic")

        settings = {"asr": asr, "tts": tts, "judge": judge, "utmos": utmos, "denoise": denoise,
                    "embed": embed_mode, "model_id": model_id, "mfa": enable_mfa,
                    "dictionaries": dictionaries, "acoustics": acoustics, "use_gpu": use_gpu,
                    "metadata": edited.to_dict("records")}
        audio_bytes = b"".join(upload.getvalue() for upload in uploads)
        digest = _settings_digest(audio_bytes, "", settings)

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
                if language not in LANGUAGES:
                    errors.append({"file": name, "error": f"Unknown language '{language}' (use hi, hi-en or sa)"})
                    progress.progress((idx + 1) / len(uploads))
                    continue
                if not normalize_reference_text(text):
                    errors.append({"file": name, "error": "Missing golden-reference transcript"})
                    progress.progress((idx + 1) / len(uploads))
                    continue
                try:
                    audio, sr, duration, info = decode_audio(upload.getvalue(), Path(name).suffix, 120, return_info=True)
                    options = {"asr": asr, "tts": tts, "tts_source": "reference", "judge": judge,
                               "sarvam_mode": "codemix", "denoise": denoise, "utmos": utmos, "release": True,
                               "language": language, "clipping_fraction": info["clipping_fraction"],
                               "use_gpu": use_gpu, "defer_maya_release": "maya2-native" in tts}
                    with st.status(f"{name}: running models", expanded=False) as status:
                        result = run_research(audio, sr, text, options, status)
                    result["name"] = name
                    result["run_id"] = f"{digest[:12]}::{name}"
                    result["speaker"] = str(meta.get("speaker") or Path(name).stem)
                    result["provenance"] = str(meta.get("provenance") or "unknown")
                    results.append(result)
                    if embed_mode != "Off":
                        vector = (acoustic_embedding(audio, sr) if embed_mode.startswith("MFCC")
                                  else wav2vec_embedding(audio, sr, model_id,
                                                         str(_inference_device(use_gpu))))
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
            maya_label = TTS_ENGINES["maya2-native"]["label"]
            maya_loaded = any(
                row.get("Step") == f"Load · {maya_label}" and row.get("Status") == "ok"
                for result in results
                for row in result["log"]
            )
            if maya_loaded:
                release("maya2-native")
            st.session_state["batch_results"] = (results, embeddings, alignment_rows, errors)
            st.session_state["batch_digest"] = digest
        stored = st.session_state.get("batch_results")
        if stored is None:
            return
        if st.session_state.get("batch_digest") != digest:
            st.warning("Inputs or settings changed after the last run. The results below are from that earlier run; run the batch again to update them.")
        results, embeddings, alignment_rows, errors = stored
        if errors:
            st.warning("Some files could not be processed.")
            st.dataframe(pd.DataFrame(errors), hide_index=True)
        if not results:
            return

        scores = _asr_rows(results)
        st.subheader("Across-file ASR benchmark")
        st.dataframe(scores, hide_index=True, width="stretch")
        st.download_button("Download ASR results CSV", scores.to_csv(index=False).encode(), "batch_asr.csv", "text/csv")
        _render_corpus_summary(scores)

        tts_table = _tts_rows(results)
        if not tts_table.empty:
            _render_tts_summary(results, tts_table)

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
            st.caption("Boundaries come from forced alignment against the reference text; no manual boundary ground truth is checked.")
        st.subheader("Individual recording results")
        selected = st.selectbox("Inspect file", [r["name"] for r in results], key="batch_inspect")
        result = next(r for r in results if r["name"] == selected)
        render_result_tabs(result)
    except Exception as exc:
        st.error(f"Batch setup error: {type(exc).__name__}: {exc}")


def _asr_rows(results) -> pd.DataFrame:
    rows = []
    for result in results:
        for row in result["asr"]:
            rows.append({
                "file": result["name"], "language": result["language"], "provenance": result.get("provenance", "unknown"),
                "model_key": row["key"], "model": row["label"], "status": row["status"],
                "primary metric": PRIMARY_LABEL.get(PRIMARY_METRIC.get(result["language"], "wer")),
                "primary (%)": row.get("primary"), "WER norm (%)": row.get("wer"), "WER raw (%)": row.get("wer_raw"),
                "WER script-tol. (%)": row.get("wer_script_tolerant"), "CER norm (%)": row.get("cer"),
                "errors_norm": row.get("errors_norm"), "errors_script_tolerant": row.get("errors_script_tolerant"),
                "reference_words_norm": row.get("reference_words_norm"), "char_errors_norm": row.get("char_errors_norm"),
                "reference_chars_norm": row.get("reference_chars_norm"),
                "Latency (s)": row.get("latency"), "RTF": row.get("rtf"), "Load (s)": row.get("load_seconds"),
                "note": row.get("error") if row["status"] != "ok" else row.get("note", ""),
            })
    return pd.DataFrame(rows)


MIN_FILES_FOR_CI = 4  # resampling 2-3 files gives ranges too coarse to interpret


def _verdict(row) -> str:
    if not np.isfinite(row.get("ci_low", np.nan)) or not np.isfinite(row.get("ci_high", np.nan)):
        return "not enough files"
    if row["ci_high"] < 0:
        return f"{row['model_a']} better"
    if row["ci_low"] > 0:
        return f"{row['model_b']} better"
    return "no clear difference"


def _render_corpus_summary(scores: pd.DataFrame) -> None:
    """Pooled corpus metric per variety. Below MIN_FILES_FOR_CI files: error counts per file
    (the real data) instead of empty CI columns. From MIN_FILES_FOR_CI files: file-bootstrap 95%
    CIs and paired model tests with a plain verdict. Varieties are never pooled; unsupported
    model-language pairs are dropped before the common-file rule."""
    st.subheader("Corpus summary")
    include_unknown = st.toggle(
        "Include files whose reference provenance is 'unknown'", value=True, key="batch_include_unknown",
        help="Turn off for thesis headline numbers once each file's provenance is set.")
    frame = scores[scores.status != "unsupported"].copy()
    if not include_unknown:
        frame = frame[frame.provenance != "unknown"]
    if frame.empty:
        st.info("No scored files for the summary (check the provenance filter).")
        return
    for language in [lang for lang in LANGUAGES if lang in set(frame.language)]:
        error_col, denom_col = CORPUS_COLUMNS[language]
        unit = "characters" if language == "sa" else "words"
        label = PRIMARY_LABEL[PRIMARY_METRIC[language]]
        summary, _, pairs, excluded = stats.fair_asr_comparison(
            frame, language, error_col, denom_col, model_col="model", status_col="status",
            file_col="file", language_col="language")
        st.markdown(f"#### {LANGUAGES[language]} — {label}")
        if summary.empty:
            st.info("No common set of files where every selected model produced a scored transcript.")
            _show_excluded(excluded)
            continue
        n_files = int(summary.files.min())
        common = frame[(frame.language == language) & (frame.status == "ok")
                       & frame.model.isin(summary.model)]
        common = common[common.file.isin(_common_files(common, summary.model))]
        errors = (common.groupby("model")[error_col].sum().astype(int).rename(f"wrong {unit}"))
        table = pd.DataFrame({
            "model": summary.model,
            "files": summary.files,
            f"reference {unit}": summary.reference_words,
            f"wrong {unit}": summary.model.map(errors),
            f"{label} (%)": summary.corpus_wer * 100,
        })
        if n_files >= MIN_FILES_FOR_CI:
            table["95% CI low (%)"] = summary.corpus_ci_low * 100
            table["95% CI high (%)"] = summary.corpus_ci_high * 100
            table["per-file mean (%)"] = summary.macro_wer * 100
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.2f") for c in table.columns if c.endswith("(%)")
        })
        if float(summary.largest_file_word_share.max()) > 0.5:
            st.warning("One file supplies most of the reference text; these numbers mostly describe that recording.")

        per_file = common.pivot_table(index="file", columns="model", values=error_col, aggfunc="sum")
        per_file = per_file[list(summary.model)].astype(int)
        per_file.insert(0, f"reference {unit}", common.groupby("file")[denom_col].first().astype(int))
        if n_files < MIN_FILES_FOR_CI:
            st.markdown(f"**Wrong {unit} per file**")
            st.dataframe(per_file.reset_index(), hide_index=True, width="stretch")
            st.info(f"{n_files} file(s) of {LANGUAGES[language]} so far. Confidence ranges and model-vs-model "
                    f"tests appear from {MIN_FILES_FOR_CI} files — add {MIN_FILES_FOR_CI - n_files} more. "
                    "Until then, a difference of one or two wrong words is not evidence that one model is better.")
        else:
            shown = pairs.copy()
            shown["verdict"] = shown.apply(_verdict, axis=1)
            for column in ("delta", "ci_low", "ci_high"):
                shown[column] = shown[column] * 100
            shown["p_a_better"] = shown["p_a_better"] * 100
            shown = shown.rename(columns={
                "model_a": "model A", "model_b": "model B", "delta": "A − B (points)",
                "ci_low": "95% CI low", "ci_high": "95% CI high", "p_a_better": "A better in (%) of resamples"})
            st.markdown("**Model vs model** (a difference is clear only when its whole 95% range is on one side of zero)")
            st.dataframe(shown, hide_index=True, width="stretch", column_config={
                c: st.column_config.NumberColumn(format="%.2f")
                for c in ("A − B (points)", "95% CI low", "95% CI high", "A better in (%) of resamples")
            })
            with st.expander(f"Wrong {unit} per file"):
                st.dataframe(per_file.reset_index(), hide_index=True, width="stretch")
        _show_excluded(excluded)
    st.caption("Script-tolerant WER is the headline for code-mixed speech, CER for Sanskrit. "
               "Confidence ranges resample whole files (file-level bootstrap, 2,000 draws).")


def _common_files(frame: pd.DataFrame, models) -> list:
    counts = frame.groupby("file").model.nunique()
    return list(counts[counts == len(set(models))].index)


def _show_excluded(excluded: pd.DataFrame) -> None:
    if len(excluded):
        with st.expander("Files left out of this block, with reasons"):
            st.dataframe(excluded, hide_index=True, width="stretch")


def _tts_rows(results) -> pd.DataFrame:
    rows = []
    for result in results:
        for row in result["tts"]:
            loop = row.get("loopback", {})
            raw_f0 = result.get("prosody", {}).get(row["label"], {}).get("metrics", {}).get("F0 median (Hz)")
            try:
                median_f0 = float(raw_f0)
            except (TypeError, ValueError):
                median_f0 = np.nan
            rows.append({
                "file": result["name"], "language": result["language"], "engine_key": row["key"], "engine": row["label"],
                "status": row["status"], "loopback (%)": loop.get("primary"), "loopback WER norm (%)": loop.get("wer"),
                "loopback CER (%)": loop.get("cer"), "UTMOS": row.get("utmos"), "RTF": row.get("rtf"),
                "Median F0 (Hz)": median_f0,
                "note": row.get("error") if row["status"] != "ok" else row.get("note", ""),
            })
    return pd.DataFrame(rows)


def _render_tts_summary(results, table: pd.DataFrame) -> None:
    st.subheader("Across-file TTS benchmark")
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download TTS results CSV", table.to_csv(index=False).encode(), "batch_tts.csv", "text/csv")
    ok = table[table.status == "ok"].copy()
    if ok.empty:
        return
    ok["higher-pitch candidate"] = (
        pd.to_numeric(ok["Median F0 (Hz)"], errors="coerce") >= FEMALE_VOICE_F0_THRESHOLD_HZ
    )
    for column in ("loopback (%)", "UTMOS", "RTF"):
        ok[column] = pd.to_numeric(ok[column], errors="coerce")
    summary = ok.groupby(["language", "engine"]).agg(
        files=("file", "nunique"), higher_pitch_files=("higher-pitch candidate", "sum"),
        loopback_mean=("loopback (%)", "mean"), utmos_mean=("UTMOS", "mean"), rtf_mean=("RTF", "mean"),
    ).reset_index()
    summary["ranked"] = summary.higher_pitch_files == summary.files
    st.markdown(
        "**Per-engine means** — an engine is ranked only when every successful file has measurable "
        f"median F0 at or above {FEMALE_VOICE_F0_THRESHOLD_HZ:.0f} Hz."
    )
    st.caption("F0 is a rough voice-range heuristic, not a gender test.")
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.3f") for c in ("loopback_mean", "utmos_mean", "rtf_mean")
    })
    if not summary.ranked.any():
        st.info(
            "No engine has measurable median F0 above the threshold for every successful file, "
            "so no TTS ranking is claimed."
        )
    if ok["loopback (%)"].isna().all():
        st.caption("Loopback is empty: choose a loopback judge above to measure intelligibility.")


def _embedding_npz(items):
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"embedding_{i}": r["embedding"] for i, r in enumerate(items)},
                        filenames=np.asarray([r["file"] for r in items]))
    return buf.getvalue()
