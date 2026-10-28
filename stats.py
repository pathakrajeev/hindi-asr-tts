"""Bootstrap intervals and fair model comparison (copied unchanged from the notebook package cmhss/stats.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_ratio_ci(errors, denominators, repetitions=2000, seed=42, alpha=0.05):
    """File-level (cluster) bootstrap CI for a pooled ratio such as corpus WER."""
    e = np.asarray(errors, float)
    n = np.asarray(denominators, float)
    ok = np.isfinite(e) & np.isfinite(n) & (n > 0)
    e, n = e[ok], n[ok]
    if len(e) == 0:
        return dict(value=np.nan, ci_low=np.nan, ci_high=np.nan, files=0)
    value = e.sum() / n.sum()
    if len(e) < 2:
        return dict(value=float(value), ci_low=np.nan, ci_high=np.nan, files=len(e))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e), size=(repetitions, len(e)))
    draws = e[idx].sum(1) / np.maximum(n[idx].sum(1), 1e-12)
    return dict(value=float(value), ci_low=float(np.quantile(draws, alpha / 2)),
                ci_high=float(np.quantile(draws, 1 - alpha / 2)), files=int(len(e)))


def bootstrap_mean_ci(values, repetitions=2000, seed=42, alpha=0.05):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(mean=np.nan, ci_low=np.nan, ci_high=np.nan, n=0)
    if len(v) < 2:
        return dict(mean=float(v.mean()), ci_low=np.nan, ci_high=np.nan, n=1)
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), size=(repetitions, len(v)))].mean(1)
    return dict(mean=float(v.mean()), ci_low=float(np.quantile(draws, alpha / 2)),
                ci_high=float(np.quantile(draws, 1 - alpha / 2)), n=int(len(v)))


def paired_bootstrap_delta(frame, model_a, model_b, error_col, denom_col, repetitions=2000, seed=42):
    """P(model_a corpus error < model_b) and CI of the difference, resampling files."""
    a = frame[frame.model == model_a].set_index("file")
    b = frame[frame.model == model_b].set_index("file")
    files = sorted(set(a.index) & set(b.index))
    if len(files) < 2:
        return dict(model_a=model_a, model_b=model_b, files=len(files), delta=np.nan,
                    ci_low=np.nan, ci_high=np.nan, p_a_better=np.nan)
    ea, eb = a.loc[files, error_col].to_numpy(float), b.loc[files, error_col].to_numpy(float)
    n = a.loc[files, denom_col].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(files), size=(repetitions, len(files)))
    d = ea[idx].sum(1) / n[idx].sum(1) - eb[idx].sum(1) / n[idx].sum(1)
    return dict(model_a=model_a, model_b=model_b, files=len(files),
                delta=float(ea.sum() / n.sum() - eb.sum() / n.sum()),
                ci_low=float(np.quantile(d, .025)), ci_high=float(np.quantile(d, .975)),
                p_a_better=float((d < 0).mean()))


def fair_asr_comparison(frame, language="hi", error_col="errors_norm", denom_col="reference_words_norm",
                        model_col="model", status_col="status", file_col="file", language_col="language"):
    """Corpus + macro WER on the intersection of files where EVERY model scored."""
    f = frame.rename(columns={model_col: "model", file_col: "file"}).copy()
    if language_col in f:
        other = f[f[language_col].astype(str) != language]
        f = f[f[language_col].astype(str) == language]
    else:
        other = f.iloc[0:0]
    models = sorted(f.model.unique())
    ok = f[(f[status_col] == "ok") & np.isfinite(pd.to_numeric(f[error_col], errors="coerce"))
           & (pd.to_numeric(f[denom_col], errors="coerce") > 0)]
    counts = ok.groupby("file").model.nunique()
    common = sorted(counts[counts == len(models)].index)
    excluded = [{"file": fl, "reason": "not all models produced a scored transcript"}
                for fl in sorted(set(f.file) - set(common))]
    excluded += [{"file": fl, "reason": f"language {lg} != {language}"}
                 for fl, lg in other[[ "file", language_col]].drop_duplicates().itertuples(index=False)]
    data = ok[ok.file.isin(common)].copy()
    data[error_col] = data[error_col].astype(float)
    data[denom_col] = data[denom_col].astype(float)
    rows = []
    for m, g in data.groupby("model"):
        corpus = bootstrap_ratio_ci(g[error_col], g[denom_col])
        macro = bootstrap_mean_ci(g[error_col] / g[denom_col])
        rows.append({"model": m, "files": corpus["files"], "reference_words": int(g[denom_col].sum()),
                     "corpus_wer": corpus["value"], "corpus_ci_low": corpus["ci_low"],
                     "corpus_ci_high": corpus["ci_high"], "macro_wer": macro["mean"],
                     "macro_ci_low": macro["ci_low"], "macro_ci_high": macro["ci_high"],
                     "largest_file_word_share": float(g[denom_col].max() / g[denom_col].sum())})
    summary = pd.DataFrame(rows).sort_values("corpus_wer") if rows else pd.DataFrame()
    pairs = []
    ms = list(summary.model) if len(summary) else []
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            pairs.append(paired_bootstrap_delta(data, ms[i], ms[j], error_col, denom_col))
    per_file = data.pivot_table(index="file", columns="model", values=error_col, aggfunc="sum")
    per_file = per_file.div(data.groupby("file")[denom_col].first(), axis=0)
    return summary, per_file.reset_index(), pd.DataFrame(pairs), pd.DataFrame(excluded)


def summarize_mushra(ratings, hidden_reference="reference", anchor="anchor", reject_threshold=90,
                     reject_fraction=0.15):
    """ITU-R BS.1534-3 style post-screening and per-system means with CIs."""
    need = {"listener_id", "trial_id", "system", "score"}
    if not need.issubset(ratings.columns):
        raise ValueError(f"MUSHRA ratings need columns {sorted(need)}")
    r = ratings.copy()
    r["score"] = pd.to_numeric(r.score, errors="raise")
    if not r.score.between(0, 100).all():
        raise ValueError("MUSHRA scores must be 0-100")
    ref = r[r.system == hidden_reference]
    bad = ref.assign(low=ref.score < reject_threshold).groupby("listener_id").low.mean()
    rejected = sorted(bad[bad > reject_fraction].index)
    kept = r[~r.listener_id.isin(rejected)]
    per_listener = kept.groupby(["system", "listener_id"]).score.mean().reset_index()
    rows = [{"system": s, **bootstrap_mean_ci(g.score)} for s, g in per_listener.groupby("system")]
    return pd.DataFrame(rows).sort_values("mean", ascending=False), rejected