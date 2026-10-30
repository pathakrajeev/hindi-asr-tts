"""Naturalness mode: run the human-likeness pipeline, inspect results, and run the listening test.

Every step calls naturalness.py, the same code the CLI (naturalness_cli.py) uses.
"""
from __future__ import annotations

import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import naturalness as nat

SECTIONS = {
    "pipeline": ":material/account_tree: Pipeline",
    "results": ":material/query_stats: Results",
    "listen": ":material/headphones: Listening test",
    "listen_results": ":material/how_to_vote: Listening results",
}


def _status_table(corpus: nat.Corpus) -> pd.DataFrame:
    sources = [nat.HUMAN, *corpus.systems()]
    texts = corpus.text_of()
    rows = []
    for source in sources:
        for speaker in corpus.speakers():
            audio = sum(corpus.audio(source, speaker, sid) is not None for sid in texts)
            grids = sum(corpus.textgrid(source, speaker, sid).is_file() for sid in texts)
            rows.append({"source": source, "speaker": speaker, "audio files": audio, "aligned": grids,
                         "stimuli": len(texts)})
    return pd.DataFrame(rows)


def _run(label: str, fn):
    """Run one pipeline step with its log lines streamed into a status box."""
    with st.status(label, expanded=True) as status:
        try:
            result = fn(lambda message: status.write(str(message)))
            failures = result.get("failed", []) if isinstance(result, dict) else []
            if failures:
                status.update(label=f"{label}: partial", state="error")
                st.warning(f"{len(failures)} file(s) failed. See the filenames below.")
                st.code("\n".join(str(path) for path in failures), language="text")
            else:
                status.update(label=f"{label}: done", state="complete")
            return result
        except Exception as exc:
            status.update(label=f"{label}: failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            return None


def _render_pipeline(corpus: nat.Corpus) -> None:
    if not corpus.stimuli_csv.is_file():
        st.info("Create the corpus first, then add recordings to the folders it creates.",
                title="No corpus at this folder")
        with st.form("nat_create_corpus"):
            sentences_text = st.text_area(
                "Pilot stimuli (one sentence per line)",
                help="Start with a small pilot. Each line becomes one stimulus; keep this wording fixed after splitting.",
                height=180,
            )
            speaker_ids = st.text_input("Speaker IDs", value="S01", help="Separate multiple IDs with commas.")
            submitted = st.form_submit_button("Create corpus and fixed splits", type="primary")
        if submitted:
            try:
                sentences = [line.strip() for line in sentences_text.splitlines() if line.strip()]
                speakers = [value.strip() for value in speaker_ids.split(",") if value.strip()]
                if not speakers:
                    raise ValueError("Enter at least one speaker ID.")
                created = nat.init_corpus(corpus.root, sentences, speakers)
                split = nat.make_splits(created)
                st.success(
                    f"Created {len(created.stimuli())} stimuli with fixed splits: "
                    f"{split.split.value_counts().to_dict()}."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Could not create corpus: {type(exc).__name__}: {exc}")
        return

    with st.expander("Import pilot recordings"):
        st.caption(
            "Upload audio named `S01__sent_001.wav` for human stimulus recordings, "
            "`S01__passage.wav` for the separate cloning passage, and matching "
            "`S01__passage.txt` transcript files. Existing files are never overwritten."
        )
        audio_uploads = st.file_uploader(
            "Human recordings and read-aloud cloning passages",
            type=[suffix.lstrip(".") for suffix in nat.AUDIO_SUFFIXES],
            accept_multiple_files=True,
            key="nat_recordings_upload",
        )
        transcript_uploads = st.file_uploader(
            "Read-aloud passage transcripts (.txt)",
            type=["txt"],
            accept_multiple_files=True,
            key="nat_passage_text_upload",
        )
        if st.button("Import recordings", key="nat_import_recordings", disabled=not audio_uploads):
            try:
                counts = nat.import_recordings(corpus, audio_uploads, transcript_uploads or [])
                st.success(
                    f"Imported {counts['stimuli']} human stimulus recording(s) and "
                    f"{counts['passages']} cloning passage(s)."
                )
            except Exception as exc:
                st.error(f"Recording import failed: {type(exc).__name__}: {exc}")

    st.markdown("#### Corpus check")
    for issue in nat.validate_corpus(corpus):
        {"error": st.error, "warning": st.warning, "ok": st.success}[issue["level"]](issue["message"])
    st.dataframe(_status_table(corpus), hide_index=True, alt="Audio and alignment counts per source and speaker")

    systems = corpus.systems()
    st.markdown("#### Steps")
    st.caption("Same order as NATURALNESS.md. Long steps (generation, MFA) are faster from the CLI.")

    with st.container(border=True):
        st.markdown("**1 · Generate machine speech**")
        system = st.selectbox("TTS system", [*nat.STOCK_SYSTEMS, *nat.CLONE_SYSTEMS], key="nat_gen_system")
        speaker_consent = False
        if system in nat.CLONE_SYSTEMS:
            speaker_consent = st.checkbox(
                "I have explicit permission from every speaker to clone and generate their voice.",
                key="nat_clone_consent",
            )
            st.caption("Use only the separate read-aloud passage for cloning; never use the test recordings.")
        can_generate = system not in nat.CLONE_SYSTEMS or speaker_consent
        if st.button(
            "Generate all stimuli",
            key="nat_gen",
            icon=":material/record_voice_over:",
            disabled=not can_generate,
        ):
            _run(
                f"Generating {system}",
                lambda log: nat.generate(corpus, system, speaker_consent=speaker_consent, log=log),
            )

    with st.container(border=True):
        st.markdown("**2 · MFA word alignment**")
        sources = st.multiselect("Sources", [nat.HUMAN, nat.CLONE_REF, *systems], default=[nat.HUMAN, *systems],
                                 key="nat_align_sources")
        word_lexicon = st.text_input(
            "Optional supplemental word lexicon path",
            help="UTF-8 lines: word<TAB>phone phone. Use acoustic-model phone symbols for Latin/code-mixed words.",
            key="nat_word_lexicon",
        ).strip()
        force = st.toggle("Re-align files that already have a TextGrid", key="nat_align_force")
        if st.button("Align", key="nat_align", icon=":material/align_horizontal_left:", disabled=not sources):
            def align(log):
                result = nat.align_corpus(
                    corpus, sources, force=force, word_lexicon=word_lexicon or None, log=log,
                )
                log(
                    f"Aligned {result['aligned']} of {result['requested']} requested file(s); "
                    f"{result.get('skipped_existing', 0)} already aligned."
                )
                if result.get("oov_words"):
                    log(
                        f"{len(result['oov_words'])} unresolved word(s) were not in the lexicon: "
                        + ", ".join(result["oov_words"][:30])
                    )
                return result
            _run("Running MFA", align)

    with st.container(border=True):
        st.markdown("**3 · Word durations and human − machine diff**")
        if st.button("Compute durations.csv and pairs.csv", key="nat_dur", icon=":material/timer:"):
            def step(log):
                durations = nat.build_durations(corpus)
                pairs = nat.build_pairs(corpus, durations)
                log(f"{len(durations)} aligned words; {len(pairs)} human/machine word pairs")
                coverage = nat.alignment_coverage(corpus, durations=durations)
                log("Alignment and word coverage:")
                for row in coverage.to_dict("records"):
                    log(row)
                if not pairs.empty:
                    bad = durations[durations.flag != "ok"]
                    if not bad.empty:
                        log(f"{bad[['speaker', 'sentence_id']].drop_duplicates().shape[0]} sentence(s) contain flagged words and are excluded from paired scoring.")
            _run("Building duration tables", step)
        if corpus.durations_csv.is_file():
            coverage = nat.alignment_coverage(corpus)
            st.dataframe(
                coverage,
                hide_index=True,
                alt="MFA file and word coverage, including out-of-vocabulary words",
            )
            if int(coverage["OOV words"].sum()) > 0:
                st.warning(
                    "Some words were out of vocabulary and their sentences are excluded from duration "
                    "training/evaluation. Add pronunciations in the supplemental word lexicon and realign."
                )

    with st.container(border=True):
        st.markdown("**4 · Train and apply the duration-correction layer**")
        base = [s for s in systems if "+" not in s]
        target = st.selectbox("System to correct", base, key="nat_train_system")
        left, right = st.columns(2)
        if left.button("Train (train split, choose on dev)", key="nat_train", disabled=not target,
                       icon=":material/model_training:", width="stretch"):
            def train(log):
                model, report = nat.train_duration_model(nat.load_pairs(corpus, target))
                model.save(corpus.model_path(target), {"system": target, **model.trained_on})
                report.to_csv(corpus.root / "models" / f"{nat._safe(target)}_dev_selection.csv", index=False)
                log(f"selected word={model.word_kind}, pause={model.pause_kind}")
            _run(f"Training layer for {target}", train)
        oracle = st.toggle("Oracle (real human durations; a ceiling, not a result)", key="nat_oracle")
        if right.button("Apply to dev + test", key="nat_correct", disabled=not target,
                        icon=":material/auto_fix_high:", width="stretch"):
            _run(f"Correcting {target}", lambda log: nat.correct_system(corpus, target, oracle=oracle, log=log))
        st.caption("After applying, align the new `+dur` system (step 2) and recompute durations (step 3).")

    with st.container(border=True):
        st.markdown("**5 · Fine-tune the MMS-TTS duration predictor**")
        st.caption(
            "Uses valid `mms-tts-hin` pairs: train sentences update MMS's stochastic duration predictor, "
            "dev sentences select the checkpoint, and test sentences are untouched. The text encoder, "
            "flow, and waveform decoder stay frozen; this is not full end-to-end VITS fine-tuning."
        )
        st.warning(
            "The MMS Hindi checkpoint is licensed CC-BY-NC 4.0. Use the fine-tuned model only in ways "
            "allowed by that license."
        )
        left, right = st.columns(2)
        epochs = left.number_input("Maximum epochs", min_value=1, max_value=200, value=50, step=1,
                                   key="nat_mms_ft_epochs")
        learning_rate = right.number_input(
            "Learning rate", min_value=0.000001, max_value=0.001, value=0.00001, step=0.000001,
            format="%.6f", key="nat_mms_ft_lr",
        )
        if st.button(
            "Fine-tune MMS duration predictor",
            key="nat_mms_ft",
            icon=":material/model_training:",
            disabled="mms-tts-hin" not in systems,
        ):
            import mms_finetune

            _run(
                "Fine-tuning MMS-TTS",
                lambda log: mms_finetune.fine_tune_duration_predictor(
                    corpus,
                    epochs=int(epochs),
                    learning_rate=float(learning_rate),
                    log=log,
                ),
            )
        st.caption(
            "After training, generate `mms-tts-hin-ft`, align it, rebuild durations, and evaluate it on test."
        )

    with st.container(border=True):
        st.markdown("**6 · Evaluate and build the listening test**")
        chosen = st.multiselect("Systems", systems, default=systems, key="nat_eval_systems")
        acoustic = st.toggle("Acoustic metrics (F0, speaker similarity, UTMOS, detector; downloads models)",
                             key="nat_eval_acoustic")
        left, right = st.columns(2)
        if left.button("Evaluate on test", key="nat_eval", disabled=not chosen, icon=":material/analytics:",
                       width="stretch"):
            _run("Evaluating", lambda log: nat.evaluate(corpus, chosen, "test", acoustic=acoustic, log=log))
        if right.button("Build listening kit", key="nat_kit", disabled=not chosen, icon=":material/headphones:",
                        width="stretch"):
            _run("Building listening test", lambda log: log(f"{len(nat.build_listening_test(corpus, chosen))} trials"))


def _render_results(corpus: nat.Corpus) -> None:
    report_path = corpus.root / "eval" / "report_test.csv"
    if report_path.is_file():
        report = pd.read_csv(report_path)
        st.markdown("#### Test-set report")
        st.caption("Lower word/pause RMSE and higher rhythm correlation = timing closer to the human. "
                   "Rows marked oracle use real human durations and are a ceiling, not a result.")
        st.dataframe(report, hide_index=True, alt="Objective metrics per system on the test split")
        chart = alt.Chart(report).mark_bar().encode(
            x=alt.X("word_rmse_ms:Q", title="Word-duration RMSE vs human (ms)"),
            y=alt.Y("system:N", sort="x", title=None),
            color=alt.Color("oracle:N", legend=alt.Legend(title="Oracle")),
            tooltip=["system", alt.Tooltip("word_rmse_ms:Q", format=".1f"), alt.Tooltip("rhythm_corr:Q", format=".3f")])
        st.altair_chart(chart, alt="Word-duration error per system")
        st.download_button("Download report CSV", report_path.read_bytes(), report_path.name, "text/csv",
                           icon=":material/download:")
    else:
        st.info("Run Evaluate in the Pipeline section first.", title="No evaluation report yet")

    if not corpus.pairs_csv.is_file():
        return
    pairs = nat.load_pairs(corpus)
    if pairs.empty:
        st.info("No valid human/TTS word pairs yet. Review coverage, alignments, and OOV words first.")
        return
    st.markdown("#### Human vs machine word durations")
    system = st.selectbox("System", sorted(pairs.system.unique()), key="nat_res_system")
    split = st.segmented_control("Split", list(nat.SPLITS), default="test", key="nat_res_split")
    view = pairs[(pairs.system == system) & (pairs.split == (split or "test"))]
    if view.empty:
        st.caption("No pairs for this system and split.")
        return
    a, b, c = st.columns(3)
    a.metric("Words", f"{len(view):,}", border=True)
    b.metric("Mean diff (human − machine)", f"{1000 * view.diff_s.mean():+.0f} ms", border=True)
    c.metric("Mean |diff|", f"{1000 * view.diff_s.abs().mean():.0f} ms", border=True)
    top = float(max(view.dur_s_h.max(), view.dur_s_m.max()))
    scatter = alt.Chart(view).mark_circle(size=40, opacity=0.6).encode(
        x=alt.X("dur_s_m:Q", title="Machine word duration (s)", scale=alt.Scale(domain=[0, top])),
        y=alt.Y("dur_s_h:Q", title="Human word duration (s)", scale=alt.Scale(domain=[0, top])),
        tooltip=["sentence_id", "word", alt.Tooltip("dur_s_h:Q", format=".3f"), alt.Tooltip("dur_s_m:Q", format=".3f")])
    diagonal = alt.Chart(pd.DataFrame({"x": [0, top], "y": [0, top]})).mark_line(strokeDash=[4, 4], color="gray").encode(
        x="x:Q", y="y:Q")
    st.altair_chart(scatter + diagonal, alt="Scatter of human against machine duration for each word; "
                                            "points on the dashed line match exactly")
    st.dataframe(view[["speaker", "sentence_id", "word_idx", "word", "dur_s_h", "dur_s_m", "diff_s",
                       "pause_after_s_h", "pause_after_s_m"]], hide_index=True, alt="Word-level duration pairs")

    selection = corpus.root / "models" / f"{nat._safe(system)}_dev_selection.csv"
    if selection.is_file():
        st.markdown("#### Correction-layer model selection (dev split)")
        st.dataframe(pd.read_csv(selection), hide_index=True, alt="Dev-set error for each candidate model")


def _render_listening(corpus: nat.Corpus) -> None:
    trials_path = corpus.root / "listening" / "trials.csv"
    if not trials_path.is_file():
        st.info("Build the listening kit in the Pipeline section first.", title="No listening test yet")
        return
    trials = pd.read_csv(trials_path, dtype=str)
    listener = st.session_state.get("nat_listener")
    if not listener:
        with st.form("nat_listener_form"):
            st.markdown("Each trial plays two recordings of the same sentence. **One is a real person, one is "
                        "computer-generated.** Listen to both, then choose the real person. Use headphones in a "
                        "quiet room.")
            name = st.text_input("Listener ID", placeholder="e.g. L07")
            consent = st.checkbox("I agree that my anonymous answers are stored for this research study.")
            if st.form_submit_button("Start", type="primary") and name.strip() and consent:
                st.session_state.nat_listener = name.strip()
                st.rerun()
        return

    mine = nat.listener_trials(trials, listener)
    responses = corpus.root / "listening" / "responses.csv"
    done = set()
    if responses.is_file():
        previous = pd.read_csv(responses, dtype=str)
        done = set(previous[previous.listener == listener].trial_id)
    remaining = mine[~mine.trial_id.isin(done)]
    finished = len(mine) - len(remaining)
    st.progress(finished / max(len(mine), 1), text=f"Listener {listener}: {finished} of {len(mine)} done")
    if remaining.empty:
        st.success("All trials done. Thank you!", title="Finished")
        if st.button("Next listener", icon=":material/person_add:"):
            st.session_state.pop("nat_listener", None)
            st.rerun()
        return
    trial = remaining.iloc[0]
    if st.session_state.get("nat_trial") != trial.trial_id:
        st.session_state.nat_trial, st.session_state.nat_trial_t0 = trial.trial_id, time.time()
    clips = corpus.root / "listening" / "clips"
    with st.form(f"nat_trial_{trial.trial_id}"):
        left, right = st.columns(2)
        with left:
            st.markdown("**Recording A**")
            st.audio(str(clips / trial.clip_a), alt="Recording A")
        with right:
            st.markdown("**Recording B**")
            st.audio(str(clips / trial.clip_b), alt="Recording B")
        answer = st.segmented_control("Which recording is the real person?", ["A", "B"], key=f"ans_{trial.trial_id}")
        if st.form_submit_button("Submit and continue", type="primary"):
            if answer is None:
                st.warning("Choose A or B first.")
            else:
                nat.record_response(corpus, {
                    "time": pd.Timestamp.now().isoformat(timespec="seconds"), "listener": listener,
                    "trial_id": trial.trial_id, "system": trial.system, "sentence_id": trial.sentence_id,
                    "speaker": trial.speaker, "answer": answer, "human_is": trial.human_is,
                    "correct": answer == trial.human_is,
                    "response_s": round(time.time() - st.session_state.nat_trial_t0, 1)})
                st.rerun()


def _render_listening_results(corpus: nat.Corpus) -> None:
    report, per_listener = nat.listening_report(corpus)
    if report.empty:
        st.info("No listening-test responses yet.", title="Nothing to analyse")
        return
    st.caption("Share of trials where listeners picked the real human. 50% means they could not tell human "
               "from machine (the goal); the bar shows the Wilson 95% interval.")
    base = alt.Chart(report).encode(y=alt.Y("system:N", title=None))
    chart = (base.mark_rule().encode(x=alt.X("ci_low:Q", title="Human correctly identified",
                                             scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")), x2="ci_high:Q")
             + base.mark_point(filled=True, size=80).encode(x="human_identified:Q", tooltip=list(report.columns))
             + alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="x:Q"))
    st.altair_chart(chart, alt="Human-identification rate per system with 95% intervals; dashed line marks chance")
    st.dataframe(report, hide_index=True, alt="Listening-test results per system")
    with st.expander("Per listener"):
        st.dataframe(per_listener, hide_index=True, alt="Accuracy per listener and system")
    responses = corpus.root / "listening" / "responses.csv"
    st.download_button("Download responses CSV", responses.read_bytes(), "responses.csv", "text/csv",
                       icon=":material/download:")
    st.caption("Hugging Face Spaces storage is not persistent: download responses after each session.")


def render_naturalness_mode() -> None:
    st.subheader("Naturalness · human vs machine timing")
    root = st.text_input("Corpus folder", value=str(nat.DEFAULT_ROOT), key="nat_root")
    corpus = nat.Corpus(Path(root))
    section = st.segmented_control("Section", list(SECTIONS), format_func=SECTIONS.get, default="pipeline",
                                   key="nat_section", label_visibility="collapsed")
    if section == "pipeline" or section is None:
        _render_pipeline(corpus)
    elif section == "results":
        _render_results(corpus)
    elif section == "listen":
        _render_listening(corpus)
    else:
        _render_listening_results(corpus)
