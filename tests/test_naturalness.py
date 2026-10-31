"""End-to-end test of the naturalness pipeline on synthetic speech, with fake_mfa.py standing in for MFA.

Human and machine "speech" render each word as a Hann-enveloped tone. The human speaker lengthens
phrase-final and content words and pauses at commas; the machine reads every syllable at one rate.
The learned layer must close that gap on held-out test sentences after re-alignment.

Run with `python tests/test_naturalness.py` or pytest.
"""
import io
import os
import shutil
import sys
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import naturalness as nat  # noqa: E402
import mms_finetune  # noqa: E402
import naturalness_ui  # noqa: E402
import voice_clone  # noqa: E402

VOCAB = ["आज", "मौसम", "बहुत", "अच्छा", "है", "हम", "बाज़ार", "जाएँगे", "क्या", "तुम", "स्कूल", "गए", "थे",
         "किताब", "मेज़", "पर", "रखी", "पानी", "ठंडा", "कल", "बारिश", "होगी", "मेरा", "नाम", "राम", "दिल्ली"]


def _sentence(rng):
    words = list(rng.choice(VOCAB, size=rng.integers(4, 9), replace=True))
    if len(words) > 5:
        words[2] += ","
    return " ".join(words) + rng.choice(["।", "?"])


def _render(text, sr, human, rng):
    tokens = nat.tokens_with_punct(text)
    pieces = [np.zeros(int(0.2 * sr))]
    for i, (word, punct) in enumerate(tokens):
        syll = nat.word_shape(word)["n_syll"]
        if human:
            dur = 0.11 * syll * (1.45 if i == len(tokens) - 1 else 1.0) * (0.8 if word in nat.FUNCTION_WORDS else 1.1)
            dur *= rng.uniform(0.95, 1.05)
            pause = 0.28 if punct == "comma" else 0.0
        else:
            dur, pause = 0.09 * syll, 0.0
        t = np.arange(int(dur * sr)) / sr
        f0 = 140 + 40 * rng.random()
        tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3))
        pieces += [0.3 * tone * np.hanning(len(t)), np.zeros(int(pause * sr))]
    pieces.append(np.zeros(int(0.2 * sr)))
    return np.concatenate(pieces).astype(np.float32)


def test_pipeline():
    rng = np.random.default_rng(0)
    work = Path(tempfile.mkdtemp(prefix="nat_test_"))
    try:
        bin_dir = work / "bin"
        bin_dir.mkdir()
        if os.name == "nt":
            (bin_dir / "fakemfa.cmd").write_text(f'@"{sys.executable}" "{ROOT / "tests" / "fake_mfa.py"}" %*\n')
        else:
            script = bin_dir / "fakemfa"
            script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{ROOT / "tests" / "fake_mfa.py"}" "$@"\n')
            script.chmod(0o755)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]
        mfa = {"command": "fakemfa", "acoustic": nat.DEFAULT_ACOUSTIC, "dictionary": nat.DEFAULT_DICTIONARY,
               "log": lambda *_: None}

        sentences = list(dict.fromkeys(_sentence(rng) for _ in range(80)))[:60]
        stim = work / "s.txt"
        stim.write_text("\n".join(sentences), encoding="utf-8")
        corpus = nat.init_corpus(work / "corpus", sentences, ["S01"])
        nat.make_splits(corpus)

        # cloning passage: leakage check must catch a copied stimulus, and pass a clean passage
        (corpus.clone_dir("S01") / "passage.txt").write_text("एक बार एक राजा था। " + sentences[0], encoding="utf-8")
        sf.write(corpus.clone_dir("S01") / "passage.wav", np.zeros(16000 * 90, np.float32), 16000)
        assert nat.has_errors(nat.validate_corpus(corpus)), "copied stimulus in passage must be an error"
        (corpus.clone_dir("S01") / "passage.txt").write_text("एक बार एक राजा था। वह जंगल गया।", encoding="utf-8")

        for sid, text in corpus.text_of().items():
            sf.write(corpus.root / "speakers" / "S01" / "stimuli" / f"{sid}.wav", _render(text, 16000, True, rng), 16000)
            target = corpus.generated_path("tts", "S01", sid)
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, _render(text, 24000, False, rng), 24000)
        assert not nat.has_errors(nat.validate_corpus(corpus))

        result = nat.align_corpus(corpus, ["human", "tts"], **mfa)
        assert result["aligned"] == 2 * len(sentences), result
        assert not result["oov_words"], result["oov_words"]
        grid = corpus.textgrid("human", "S01", "sent_001")
        words = nat.word_intervals(grid)
        assert [w["word"] for w in words] == [w for w, _ in nat.tokens_with_punct(corpus.text_of()["sent_001"])]

        durations = nat.build_durations(corpus)
        assert (durations.flag == "ok").mean() > 0.95, durations.flag.value_counts()
        coverage = nat.alignment_coverage(corpus, durations=durations).set_index("Source")
        assert coverage.loc["human", "Alignment coverage (%)"] == 100
        assert coverage.loc["tts", "OOV words"] == 0
        pairs = nat.build_pairs(corpus, durations)
        assert {"diff_s", "log_ratio", "pause_diff_s"} <= set(pairs.columns)
        assert np.allclose(pairs.diff_s, pairs.dur_s_h - pairs.dur_s_m)

        model, report = nat.train_duration_model(nat.load_pairs(corpus, "tts"))
        assert report.selected.sum() == 1
        chosen = report[report.selected].iloc[0]
        assert chosen.dev_word_rmse_ms < 0.6 * chosen.machine_word_rmse_ms, report
        model.save(corpus.model_path("tts"))

        nat.correct_system(corpus, "tts", log=lambda *_: None)
        nat.correct_system(corpus, "tts", oracle=True, log=lambda *_: None)
        nat.align_corpus(corpus, ["tts+dur", "tts+oracle"], **mfa)
        nat.build_pairs(corpus, nat.build_durations(corpus))
        report, _ = nat.evaluate(corpus, ["tts", "tts+dur", "tts+oracle"], "test", log=lambda *_: None)
        by = report.set_index("system")
        print(by[["sentences", "word_rmse_ms", "rhythm_corr", "pause_rmse_ms", "pause_f1"]].round(3).to_string())
        assert by.sentences.nunique() == 1, "every system must be scored on the same sentences"
        assert by.loc["tts+dur", "word_rmse_ms"] < 0.6 * by.loc["tts", "word_rmse_ms"]
        assert by.loc["tts+dur", "pause_rmse_ms"] < by.loc["tts", "pause_rmse_ms"]
        assert by.loc["tts+oracle", "oracle"] and not by.loc["tts+dur", "oracle"]

        trials = nat.build_listening_test(corpus, ["tts", "tts+dur", "tts+oracle"])
        assert set(trials.system) == {"tts", "tts+dur"}, "oracle must never enter the listening test"
        assert set(trials.split if "split" in trials else ["test"]) == {"test"}
        for listener in ("L1", "L2", "L3", "L4"):
            mine = nat.listener_trials(trials, listener)
            assert mine.sentence_id.is_unique
            for row in mine.itertuples():
                nat.record_response(corpus, {"listener": listener, "trial_id": row.trial_id, "system": row.system,
                                             "sentence_id": row.sentence_id, "answer": row.human_is,
                                             "correct": row.system == "tts" or rng.random() < 0.5})
        listen, _ = nat.listening_report(corpus)
        print(listen.round(3).to_string(index=False))
        assert listen.set_index("system").loc["tts", "human_identified"] == 1.0

        try:
            nat.evaluate(corpus, ["tts", "missing-system"], "test", ci=False, log=lambda *_: None)
        except ValueError as exc:
            assert "missing-system" in str(exc)
        else:
            raise AssertionError("evaluation must fail rather than silently omit a selected system")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_supplemental_lexicon_and_code_mixed_oov(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    phones = nat.phone_inventory(nat.DEFAULT_ACOUSTIC)
    lexical = tmp / "words.txt"
    lexical.write_text(f"hello\t{phones.pop()}\n", encoding="utf-8")
    overrides = nat.load_word_lexicon(lexical, phones | {lexical.read_text(encoding="utf-8").split()[1]})
    assert "hello" in overrides
    try:
        nat.load_word_lexicon(lexical, phones)
    except ValueError as exc:
        assert "absent from the acoustic model" in str(exc)
    else:
        raise AssertionError("unknown phone symbols must be rejected")

    grid = tmp / "oov.TextGrid"
    grid.write_text(
        'File type = "ooTextFile"\nObject class = "TextGrid"\n\n0\n1\n<exists>\n2\n'
        '"IntervalTier"\n"words"\n0\n1\n1\n0\n1\n"hello"\n'
        '"IntervalTier"\n"phones"\n0\n1\n1\n0\n1\n"spn"\n',
        encoding="utf-8",
    )
    assert nat.word_intervals(grid)[0]["oov"]


def test_split_validation(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    corpus = nat.init_corpus(tmp / "corpus", ["आज है।", "कल है।", "अब है।"], ["S01"])
    try:
        nat.make_splits(corpus, train=0.8, dev=0.3)
    except ValueError as exc:
        assert "sum to less than 1" in str(exc)
    else:
        raise AssertionError("invalid split fractions must be rejected")


def test_empty_pipeline_and_force_protection(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    corpus = nat.init_corpus(tmp / "corpus", ["आज है।", "कल है।", "अब है।"], ["S01"])
    empty = nat.build_pairs(corpus, pd.DataFrame())
    assert empty.empty and list(empty.columns) == nat.PAIR_COLUMNS
    assert nat.load_pairs(corpus).empty
    try:
        nat.train_duration_model(empty)
    except ValueError as exc:
        assert "no valid word pairs" in str(exc)
    else:
        raise AssertionError("training must reject empty input")

    try:
        nat.align_corpus(corpus, ["human"], command="not-run")
    except ValueError as exc:
        assert "No matching audio files" in str(exc)
    else:
        raise AssertionError("alignment must reject a corpus with no selected audio")

    nat.make_splits(corpus)
    try:
        nat.init_corpus(corpus.root, ["नया वाक्य।"], ["S01"], force=True)
    except ValueError as exc:
        assert "already has analysis data" in str(exc)
    else:
        raise AssertionError("force must not overwrite a corpus after splits are created")


def test_missing_split_recovery_guard(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    corpus = nat.init_corpus(tmp / "corpus", ["आज है।", "कल है।", "अब है।"], ["S01"])
    assert not naturalness_ui._has_analysis_artifacts(corpus)
    nat.make_splits(corpus)
    corpus.splits_csv.unlink()
    generated = corpus.generated_path("mms-tts-hin", "S01", "sent_001")
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_bytes(b"already analyzed")
    assert naturalness_ui._has_analysis_artifacts(corpus)


def test_append_stimuli_before_splits(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    corpus = nat.init_corpus(tmp / "corpus", ["आज है।"], ["S01"])
    updated = nat.append_stimuli(corpus, ["कल है।", "अब है।"])
    assert updated.sentence_id.tolist() == ["sent_001", "sent_002", "sent_003"]
    assert updated.text.tolist() == ["आज है।", "कल है।", "अब है।"]
    nat.make_splits(corpus)

    try:
        nat.append_stimuli(corpus, ["परसों है।"])
    except FileExistsError as exc:
        assert "frozen after splits.csv" in str(exc)
    else:
        raise AssertionError("stimuli must not change after fixed splits are created")

    incomplete = nat.init_corpus(tmp / "incomplete", ["आज है।"], ["S01"])
    try:
        nat.append_stimuli(incomplete, ["आज है।"])
    except ValueError as exc:
        assert "duplicates text already in the corpus" in str(exc)
    else:
        raise AssertionError("appending an existing normalized sentence must be rejected")


def test_voice_clone_worker_failures_and_model_revision(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    blocker = tmp / "not_a_directory"
    blocker.write_text("file", encoding="utf-8")
    events = []
    with patch.dict(voice_clone.LOADERS, {"test": lambda _spec: (lambda _text: (np.zeros(160), 16000))}):
        try:
            voice_clone.run_jobs(
                {"backend": "test", "device": "cpu", "jobs": [{"text": "नमस्ते", "out": str(blocker / "out.wav")}]},
                events.append,
            )
        except RuntimeError as exc:
            assert "failed for 1 of 1 file(s)" in str(exc)
        else:
            raise AssertionError("the worker must not report success when a generated file could not be written")
    assert events[-1]["event"] == "error"

    calls = {}

    class FakeModel:
        def __call__(self, _text, **_kwargs):
            return np.zeros(160, dtype=np.float32)

    def from_pretrained(model_id, **kwargs):
        calls["model_id"] = model_id
        calls.update(kwargs)
        return FakeModel()

    fake_transformers = SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=from_pretrained))
    with patch.dict(sys.modules, {"transformers": fake_transformers}):
        synth = voice_clone._indicf5(
            {"device": "cpu", "prompt_wav": "prompt.wav", "prompt_text": "नमस्ते"}
        )
        audio, sr = synth("आज मौसम अच्छा है")
    assert calls["model_id"] == "ai4bharat/IndicF5"
    assert calls["revision"] == voice_clone.INDICF5_REVISION
    assert sr == 24000 and len(audio) == 160


def test_mms_duration_finetuning(tmp_path=None):
    import torch

    tmp = Path(tmp_path or tempfile.mkdtemp())
    corpus = nat.init_corpus(
        tmp / "corpus",
        [f"आज मौसम {index} है।" for index in range(8)],
        ["S01"],
    )
    splits = pd.DataFrame({
        "sentence_id": [f"sent_{index:03d}" for index in range(1, 9)],
        "split": ["train"] * 5 + ["dev"] * 2 + ["test"],
    })
    splits.to_csv(corpus.splits_csv, index=False)
    rows = []
    for row in corpus.stimuli().itertuples():
        for word_idx, word in enumerate(nat.tokens_with_punct(row.text)):
            machine_duration = 0.10 + word_idx * 0.01
            human_duration = machine_duration + 0.04
            rows.append({
                "system": mms_finetune.BASE_SYSTEM,
                "speaker": "S01",
                "sentence_id": row.sentence_id,
                "word_idx": word_idx,
                "split": splits.set_index("sentence_id").loc[row.sentence_id, "split"],
                "n_words": len(nat.tokens_with_punct(row.text)),
                "punct_after": word[1],
                "word": word[0],
                "start_s_h": 0.0,
                "end_s_h": human_duration,
                "dur_s_h": human_duration,
                "pause_after_s_h": 0.02 if word_idx == 0 else 0.0,
                "n_phones_h": 1,
                "start_s_m": 0.0,
                "end_s_m": machine_duration,
                "dur_s_m": machine_duration,
                "pause_after_s_m": 0.0,
                "n_phones_m": 1,
                "diff_s": human_duration - machine_duration,
                "log_ratio": np.log(human_duration / machine_duration),
                "log_ratio_resid": 0.0,
                "pause_diff_s": 0.0,
            })
    pd.DataFrame(rows, columns=nat.PAIR_COLUMNS).to_csv(corpus.pairs_csv, index=False)

    class FakeTokenizer:
        add_blank = True
        is_uroman = False

        def normalize_text(self, text):
            return text

        def __call__(self, text, **kwargs):
            ids = [ord(char) for char in text for _ in (0, 1)] + [0]
            if kwargs.get("return_tensors") == "pt":
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}
            return {"input_ids": ids}

    class FakeTextEncoder(torch.nn.Module):
        def forward(self, input_ids, **_kwargs):
            hidden = torch.ones((*input_ids.shape, 2), dtype=torch.float32)
            return SimpleNamespace(last_hidden_state=hidden)

    class FakePredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, hidden, _mask, durations=None, reverse=False, **_kwargs):
            if reverse:
                return torch.log(torch.full((hidden.shape[0], 1, hidden.shape[2]), 2.0)) + 0 * self.scale
            target = durations.mean(dim=(1, 2))
            return (self.scale - target).square() + 0.1

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.duration_predictor = FakePredictor()
            self.text_encoder = FakeTextEncoder()
            self.config = SimpleNamespace(
                use_stochastic_duration_prediction=True,
                sampling_rate=16000,
                upsample_rates=[2, 2],
            )

        def forward(self, **_kwargs):
            return SimpleNamespace(waveform=torch.zeros((1, 1600), dtype=torch.float32))

    tokenizer = FakeTokenizer()
    # Verify per-character duration targets preserve the human word duration and inter-word pause.
    sentence = corpus.stimuli().iloc[0].text
    sentence_pairs = pd.read_csv(corpus.pairs_csv)
    sentence_pairs = sentence_pairs[sentence_pairs.sentence_id == "sent_001"]
    example = mms_finetune.build_duration_example(sentence, sentence_pairs, tokenizer, 100.0)
    assert len(example.input_ids) == len(example.duration_frames)
    expected_frames = sum(round(value * 100.0) for value in sentence_pairs.dur_s_h)
    expected_frames += sum(round(value * 100.0) for value in sentence_pairs.pause_after_s_h.iloc[:-1])
    assert sum(example.duration_frames) == expected_frames
    assert example.split == "train"
    prepared, _skipped = mms_finetune.prepare_examples(
        corpus, pd.concat([sentence_pairs, pd.read_csv(corpus.pairs_csv).query("split == 'test'")]),
        tokenizer, 100.0,
    )
    assert {item.split for item in prepared} <= {"train", "dev"}

    base_model = FakeModel()
    with patch.object(mms_finetune, "_load_base_model", return_value=(base_model, tokenizer)):
        metadata = mms_finetune.fine_tune_duration_predictor(
            corpus, epochs=3, learning_rate=0.01, patience=2, device="cpu", log=lambda *_: None,
        )
    assert metadata["fine_tuned_component"] == "stochastic_duration_predictor"
    assert metadata["held_out_split"] == "test"
    assert metadata["train_sentences"] == 5 and metadata["dev_sentences"] == 2
    assert metadata["best_epoch"] > 0
    assert corpus.mms_finetune_path.is_file()
    assert corpus.mms_finetune_path.with_suffix(".json").is_file()

    fake_research = SimpleNamespace(
        load_mms=lambda _device: (FakeModel(), tokenizer),
        TTS_SEED=13,
        sentence_chunks=lambda text, _limit: [text],
    )
    with patch.dict(sys.modules, {"research": fake_research}):
        audio, sample_rate = mms_finetune.synthesize(
            "आज है।", "cpu", corpus.mms_finetune_path,
        )
    assert sample_rate == 16000 and audio.shape == (1600,)

    with patch.object(mms_finetune, "synthesize", return_value=(np.zeros(1600, dtype=np.float32), 16000)):
        written = nat.generate(corpus, mms_finetune.FINETUNED_SYSTEM, splits=("test",))
    assert written == 1
    assert corpus.generated_path(mms_finetune.FINETUNED_SYSTEM, "S01", "sent_008").is_file()
    assert not corpus.generated_path(mms_finetune.BASE_SYSTEM, "S01", "sent_008").is_file()


def test_recording_import_and_clone_consent(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    corpus = nat.init_corpus(tmp / "corpus", ["आज मौसम अच्छा है।", "कल बारिश होगी।", "हम घर जाएंगे।"], ["S01"])
    nat.make_splits(corpus)
    sample = io.BytesIO()
    sf.write(sample, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    wav = sample.getvalue()
    transcript = [SimpleNamespace(name="S01__passage.txt", getvalue=lambda: b"\xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87")]
    stimulus_text = corpus.text_of()["sent_001"]
    matching_stimulus_transcript = [
        SimpleNamespace(name="S01__sent_001.txt", getvalue=lambda: stimulus_text.encode("utf-8"))
    ]
    verified_counts = nat.import_recordings(
        corpus,
        [SimpleNamespace(name="S01__sent_001.wav", getvalue=lambda: wav)],
        matching_stimulus_transcript,
    )
    assert verified_counts == {"stimuli": 1, "passages": 0, "verified_transcripts": 1}

    wrong_stimulus_transcript = [
        SimpleNamespace(name="S01__sent_002.txt", getvalue=lambda: "गलत पाठ".encode("utf-8"))
    ]
    try:
        nat.import_recordings(
            corpus,
            [SimpleNamespace(name="S01__sent_002.wav", getvalue=lambda: wav)],
            wrong_stimulus_transcript,
        )
    except ValueError as exc:
        assert "does not match the frozen stimulus text" in str(exc)
    else:
        raise AssertionError("an incorrect optional stimulus transcript must be rejected")

    counts = verified_counts
    try:
        nat.import_recordings(corpus, [], transcript)
    except ValueError as exc:
        assert "requires a matching read-aloud audio file" in str(exc)
    else:
        raise AssertionError("a cloning transcript must not be accepted without its corresponding passage audio")
    passage_counts = nat.import_recordings(
        corpus,
        [SimpleNamespace(name="S01__passage.wav", getvalue=lambda: wav)],
        transcript,
    )
    assert passage_counts == {"stimuli": 0, "passages": 1, "verified_transcripts": 0}
    try:
        nat.generate(corpus, "clone-indicf5")
    except ValueError as exc:
        assert "explicit permission" in str(exc)
    else:
        raise AssertionError("voice cloning must require explicit speaker permission")


def test_textgrid_short_format_and_pronounce(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    grid = tmp / "short.TextGrid"
    grid.write_text('File type = "ooTextFile"\nObject class = "TextGrid"\n\n0\n1.5\n<exists>\n1\n"IntervalTier"\n'
                    '"words"\n0\n1.5\n3\n0\n0.2\n""\n0.2\n0.7\n"आज"\n0.7\n1.5\n"मौसम"\n', encoding="utf-8")
    words = nat.word_intervals(grid)
    assert [(w["word"], round(w["dur_s"], 2)) for w in words] == [("आज", 0.5), ("मौसम", 0.8)]
    phones = nat.phone_inventory(nat.DEFAULT_ACOUSTIC)
    assert nat.pronounce("मौसम", phones) == ["मौ", "स", "म"]
    assert nat.pronounce("स्कूल", phones) == ["स", "्", "कू", "ल"]
    assert nat.pronounce("हैं", phones) == ["है", "ं"]
    assert nat.pronounce("बाज़ार", phones) == ["बा", "जा", "र"]
    assert nat.pronounce("hello", phones) == []


def test_leakage():
    stimuli = pd.DataFrame({"sentence_id": ["a", "b", "c"],
                            "text": ["आज का मौसम कैसा है?", "मैं घर जा रहा हूँ।", "हाँ"]})
    leaks = nat.leakage_report("कल का मौसम कैसा था। मैं घर जा रहा हूँ और", stimuli).set_index("sentence_id")
    assert not leaks.loc["a", "leak"]          # shares "का मौसम कैसा" (3 words) only
    assert leaks.loc["b", "leak"]              # whole sentence inside the passage
    assert not leaks.loc["c", "leak"]


if __name__ == "__main__":
    test_textgrid_short_format_and_pronounce()
    test_leakage()
    test_supplemental_lexicon_and_code_mixed_oov()
    test_split_validation()
    test_empty_pipeline_and_force_protection()
    test_missing_split_recovery_guard()
    test_append_stimuli_before_splits()
    test_voice_clone_worker_failures_and_model_revision()
    test_mms_duration_finetuning()
    test_recording_import_and_clone_consent()
    test_pipeline()
    print("all tests passed")
