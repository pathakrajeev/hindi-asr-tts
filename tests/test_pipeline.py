"""End-to-end test of the naturalness pipeline on synthetic speech (no real MFA, TTS or GPU needed).

    python tests/test_pipeline.py

Each "word" is a tone burst whose length follows a human-like rule (longer words, phrase-final
lengthening, pauses at commas) or a machine-like rule (near-constant length, no pauses). A fake
`mfa` finds the bursts. The test checks the five action items end to end:
MFA durations -> human-machine diff -> correction layer (train/dev only) -> corrected audio is
closer to the human on TEST -> cloning leakage is blocked -> listening kit is blind.
It does not validate real alignment quality.
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import naturalness as nat  # noqa: E402

SR = 16_000
VOCAB = ("राम सीता घर बाज़ार किताब पढ़ना स्कूल मौसम बारिश आज कल सुबह शाम खाना पानी दोस्त परिवार "
         "शहर गाँव रास्ता बहुत अच्छा सुंदर जल्दी धीरे").split()
FUNCTION = ["है", "में", "से", "को", "और", "का"]


def make_sentences(n=40, seed=3):
    rng = random.Random(seed)
    out = set()
    while len(out) < n:
        words = rng.sample(VOCAB, rng.randint(4, 7))
        words.insert(rng.randint(1, len(words) - 1), rng.choice(FUNCTION))
        if rng.random() < 0.8:
            k = rng.randint(1, len(words) - 2)
            words[k] += ","
        out.add(" ".join(words) + (" ?" if rng.random() < 0.2 else " ।"))
    return sorted(out)


def render(text, style, speaker_rate=1.0):
    """Tone bursts: one per word. Human timing depends on word length/position; machine is flat."""
    tokens = nat.tokens_with_punct(text)
    pieces = [np.zeros(int(0.2 * SR), np.float32)]
    for i, (word, punct) in enumerate(tokens):
        syll = nat.word_shape(word)["n_syll"]
        if style == "human":
            dur = (0.07 + 0.075 * syll + (0.12 if i == len(tokens) - 1 else 0.0)) * speaker_rate
            gap = 0.30 if punct == "comma" else 0.05
        else:
            dur, gap = 0.24, 0.05
        t = np.arange(int(dur * SR)) / SR
        f0 = 140 + 20 * np.sin(2 * np.pi * 0.5 * i)
        tone = 0.3 * np.sin(2 * np.pi * f0 * t) * np.hanning(len(t)) ** 0.3
        pieces += [tone.astype(np.float32), np.zeros(int(gap * SR), np.float32)]
    pieces.append(np.zeros(int(0.2 * SR), np.float32))
    return np.concatenate(pieces)


def fake_tools(folder: Path):
    script = HERE / "fake_mfa_cli.py"
    if os.name == "nt":
        mfa = folder / "mfa.cmd"
        mfa.write_text(f'@"{sys.executable}" "{script}" %*\r\n')
    else:
        mfa = folder / "mfa"
        mfa.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        mfa.chmod(0o755)
    acoustic = folder / "fake_acoustic.zip"
    with zipfile.ZipFile(acoustic, "w") as z:
        z.writestr("model/meta.json", json.dumps({"phones": [chr(c) for c in range(0x0900, 0x0980)]}))
    return str(mfa), str(acoustic)


def check(condition, message):
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        raise SystemExit(1)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mfa, acoustic = fake_tools(tmp)
        align = dict(command=mfa, acoustic=acoustic, dictionary=None, log=lambda *_: None)
        sentences = make_sentences()
        corpus = nat.init_corpus(tmp / "corpus", sentences, ["S01", "S02"])
        nat.make_splits(corpus)
        texts = corpus.text_of()

        # Human stimuli, a stock "TTS" and a clean cloning passage (no shared 4-word run).
        for spk, rate in (("S01", 1.0), ("S02", 1.15)):
            for sid, text in texts.items():
                folder = corpus.root / "speakers" / spk / "stimuli"
                sf.write(folder / f"{sid}.wav", render(text, "human", rate), SR)
                target = corpus.generated_path("fake-tts", spk, sid)
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(target, render(text, "machine"), SR)
            passage = "नदी के किनारे एक पुराना पेड़ था जिसकी छाया में बच्चे खेलते थे ।"
            sf.write(corpus.clone_dir(spk) / "passage.wav", render(passage, "human", rate), SR)
            (corpus.clone_dir(spk) / "passage.txt").write_text(passage, encoding="utf-8")
        check(not nat.has_errors(nat.validate_corpus(corpus)), "clean corpus passes the leakage check")

        # 1. MFA durations for human and machine
        result = nat.align_corpus(corpus, [nat.HUMAN, "fake-tts"], **align)
        check(result["aligned"] == 2 * 2 * len(texts) and not result["failed"], "every file aligned")
        durations = nat.build_durations(corpus)
        check((durations.flag == "ok").all(), "all aligned words pass the quality flags")

        # 2. diff = human - machine, fed to the correction layer (train/dev only)
        pairs = nat.build_pairs(corpus, durations)
        check(np.allclose(pairs.diff_s, pairs.dur_s_h - pairs.dur_s_m), "diff_s = human - machine")
        model, report = nat.train_duration_model(nat.load_pairs(corpus, "fake-tts"))
        check("none" in set(report.word_model), "no-correction baseline is a candidate")
        check(model.word_kind != "none", f"a learned layer beats no correction on dev ({model.word_kind})")
        check(set(model.trained_on["splits"]) == {"train", "dev"}, "layer never sees test")
        model.save(corpus.model_path("fake-tts"), {"system": "fake-tts"})
        nat.correct_system(corpus, "fake-tts", log=lambda *_: None)
        nat.align_corpus(corpus, ["fake-tts+dur"], **align)
        nat.build_pairs(corpus, nat.build_durations(corpus))

        # 3. evaluation on test only, same sentences for every system
        report, _ = nat.evaluate(corpus, ["fake-tts", "fake-tts+dur"], "test", ci=False, log=lambda *_: None)
        rmse = report.set_index("system").word_rmse_ms
        print(report[["system", "sentences", "word_rmse_ms", "rhythm_corr", "pause_f1"]].round(3).to_string(index=False))
        check(rmse["fake-tts+dur"] < 0.7 * rmse["fake-tts"], "corrected TTS is much closer to human timing on test")
        f1 = report.set_index("system").pause_f1
        check(f1["fake-tts+dur"] > f1["fake-tts"] + 0.3, "corrected TTS places pauses where the human did")

        # 4/5. cloning leakage: a passage containing a stimulus must block cloning
        leaky = corpus.clone_dir("S01") / "passage.txt"
        good_text = leaky.read_text(encoding="utf-8")
        leaky.write_text(good_text + " " + next(iter(texts.values())), encoding="utf-8")
        check(nat.has_errors(nat.validate_corpus(corpus)), "passage leaking a stimulus is an error")
        try:
            nat.generate(corpus, "clone-xtts", speaker_consent=True, log=lambda *_: None)
            check(False, "cloning refused on leakage")
        except RuntimeError as exc:
            check("Refusing to clone" in str(exc), "cloning refused on leakage")
        leaky.write_text(good_text, encoding="utf-8")
        (corpus.clone_dir("S01") / "prompt.txt").write_text(next(iter(texts.values())), encoding="utf-8")
        check(nat.has_errors(nat.validate_corpus(corpus)), "a prompt.txt containing a stimulus is an error")
        (corpus.clone_dir("S01") / "prompt.txt").unlink()
        try:
            nat.generate(corpus, "clone-xtts", log=lambda *_: None)
            check(False, "cloning needs speaker consent")
        except ValueError:
            check(True, "cloning needs speaker consent")

        # Listening kit is blind: clip names carry only A/B
        trials = nat.build_listening_test(corpus, ["fake-tts", "fake-tts+dur"])
        names = set(trials.clip_a) | set(trials.clip_b)
        check(len(trials) > 0 and not any("human" in n or "machine" in n for n in names),
              "listening clip names do not reveal the answer")

        # External TextGrids (another aligner): matching grids import, wrong ones are rejected
        ext = tmp / "external" / "S01"
        ext.mkdir(parents=True)
        sid = next(iter(texts))
        (ext / f"{sid}.TextGrid").write_bytes(corpus.textgrid(nat.HUMAN, "S01", sid).read_bytes())
        other = list(texts)[1]
        (ext / f"{other}.TextGrid").write_bytes(corpus.textgrid(nat.HUMAN, "S01", sid).read_bytes())
        res = nat.import_alignments(corpus, "other-aligner", tmp / "external")
        check(res["imported"] == 1 and len(res["rejected"]) == 1, "TextGrid import validates word sequence")

        # MMS duration targets: blanks share time in proportion to the base model's own durations
        import mms_finetune

        class FakeTok:
            """Like VitsTokenizer for MMS Hindi: blanks interleaved, and characters missing from the
            vocabulary (here "।", "?", ",") silently dropped."""
            add_blank = True
            missing = set("।?,")

            def normalize_text(self, text):
                return text

            def get_vocab(self):
                return {chr(c): c for c in range(0x20, 0x0980) if chr(c) not in self.missing}

            def __call__(self, text, add_special_tokens=False):
                text = "".join(c for c in text if c not in self.missing)
                return {"input_ids": [0] + [t for c in text for t in (ord(c) % 50 + 1, 0)]}

        rows = nat.load_pairs(corpus, "fake-tts")
        one = rows[(rows.speaker == "S01") & (rows.split == "train")]
        sid = one.sentence_id.iloc[0]
        group = one[one.sentence_id == sid]
        weights = lambda ids: [2.0 if t == 0 else 1.0 for t in ids]  # noqa: E731
        ex = mms_finetune.build_duration_example(texts[sid], group, FakeTok(), 86.13, weights)
        frames = np.array(ex.duration_frames)
        human_frames = sum(max(1, round(d * 86.13)) for d in group.dur_s_h) + sum(
            max(0, round(p * 86.13)) for p in group.pause_after_s_h.iloc[:-1])
        check(abs(frames.sum() - human_frames) < 1e-6, "MMS targets sum to the human word + pause frames")
        check(frames[0::2].sum() > 0, "blank tokens receive a share of the duration")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
