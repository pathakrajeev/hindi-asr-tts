"""Command-line runner for the naturalness pipeline. See NATURALNESS.md for the full protocol.

    python naturalness_cli.py init --stimuli sentences.txt --speakers S01
    python naturalness_cli.py split
    python naturalness_cli.py check
    python naturalness_cli.py generate --system bulbul-v3
    python naturalness_cli.py align --sources clone_ref
    python naturalness_cli.py import-alignments --source human --folder external_textgrids/   (optional)
    python naturalness_cli.py prompt --speaker S01
    python naturalness_cli.py generate --system clone-indicf5
    python naturalness_cli.py align --sources human bulbul-v3 clone-indicf5
    python naturalness_cli.py durations
    python naturalness_cli.py train --system clone-indicf5
    python naturalness_cli.py correct --system clone-indicf5 [--oracle]
    python naturalness_cli.py align --sources clone-indicf5+dur
    python naturalness_cli.py durations
    python naturalness_cli.py evaluate --systems bulbul-v3 clone-indicf5 clone-indicf5+dur --acoustic
    python naturalness_cli.py listening-kit --systems bulbul-v3 clone-indicf5 clone-indicf5+dur
    python naturalness_cli.py listening-report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import naturalness as nat

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def _corpus(args) -> nat.Corpus:
    return nat.Corpus(args.root)


def cmd_init(args):
    sentences = Path(args.stimuli).read_text(encoding="utf-8").splitlines()
    corpus = nat.init_corpus(args.root, sentences, args.speakers, force=args.force)
    print(f"Created {corpus.root} with {len(corpus.stimuli())} stimuli and speakers {args.speakers}")


def cmd_split(args):
    frame = nat.make_splits(_corpus(args), args.train, args.dev, args.seed, args.test_speakers, force=args.force)
    print(frame.split.value_counts().to_string())


def cmd_check(args):
    issues = nat.validate_corpus(_corpus(args))
    for issue in issues:
        print(f"[{issue['level'].upper():7}] {issue['message']}")
    if nat.has_errors(issues):
        sys.exit(1)


def cmd_generate(args):
    n = nat.generate(
        _corpus(args), args.system, tuple(args.splits), force=args.force, device=args.device,
        speaker_consent=args.speaker_consent,
    )
    print(f"{args.system}: {n} files written")


def cmd_align(args):
    result = nat.align_corpus(_corpus(args), args.sources, force=args.force, acoustic=args.acoustic,
                              dictionary=args.dictionary, word_lexicon=args.word_lexicon,
                              command=args.mfa, jobs=args.jobs)
    print(f"aligned {result['aligned']} files")
    if result.get("skipped_existing"):
        print(f"skipped {result['skipped_existing']} existing alignments")
    if result["failed"]:
        print(f"MFA produced no alignment for {len(result['failed'])}: {result['failed'][:10]}")
    if result.get("oov_words"):
        print(f"Unresolved words (these align as OOV): {', '.join(result['oov_words'][:30])}")


def cmd_convert_audio(args):
    result = nat.convert_recordings_to_wav(_corpus(args))
    print(f"converted {result['converted']} recording(s) to WAV")
    for line in result["skipped"]:
        print(f"[SKIPPED] {line}")


def cmd_import_alignments(args):
    result = nat.import_alignments(_corpus(args), args.source, args.folder, force=args.force)
    print(f"imported {result['imported']} TextGrids; skipped {result['skipped_existing']} existing")
    for line in result["rejected"]:
        print(f"[REJECTED] {line}")
    if result["rejected"] and not result["imported"]:
        sys.exit(1)


def cmd_prompt(args):
    wav, txt = nat.make_clone_prompt(_corpus(args), args.speaker)
    print(f"{wav} ({txt.read_text(encoding='utf-8')})")


def cmd_durations(args):
    corpus = _corpus(args)
    durations = nat.build_durations(corpus)
    if durations.empty:
        sys.exit("No alignments found. Run `align` first.")
    print("Word flags per source:\n" + durations.groupby(["source", "flag"]).size().unstack(fill_value=0).to_string())
    pairs = nat.build_pairs(corpus, durations)
    if pairs.empty:
        print("No human/machine pairs yet (need both human and system alignments).")
        return
    summary = pairs.groupby(["system", "split"]).agg(
        sentences=("sentence_id", "nunique"), words=("word", "size"),
        mean_diff_ms=("diff_s", lambda d: 1000 * d.mean()), mae_ms=("diff_s", lambda d: 1000 * d.abs().mean()))
    print(f"\nDuration diff (human - machine) written to {corpus.pairs_csv}:\n{summary.round(1).to_string()}")


def cmd_train(args):
    corpus = _corpus(args)
    model, report = nat.train_duration_model(nat.load_pairs(corpus, args.system), seed=args.seed)
    path = corpus.model_path(args.system)
    model.save(path, {"system": args.system, "word_model": model.word_kind, "pause_model": model.pause_kind,
                      **model.trained_on})
    report.to_csv(corpus.root / "models" / f"{nat._safe(args.system)}_dev_selection.csv", index=False)
    print(report.round(2).to_string(index=False))
    print(f"\nSelected word={model.word_kind}, pause={model.pause_kind}; saved {path}")
    if not model.trained_on.get("beats_no_correction_on_dev", True):
        print("WARNING: no candidate beat leaving the machine timing unchanged on dev. `correct` will not "
              "change the audio. Collect more train sentences or check the alignments.")


def cmd_mms_finetune(args):
    import mms_finetune

    metadata = mms_finetune.fine_tune_duration_predictor(
        _corpus(args),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    print(pd.Series(metadata).to_string())


def cmd_correct(args):
    print(nat.correct_system(_corpus(args), args.system, tuple(args.splits), oracle=args.oracle))


def cmd_evaluate(args):
    report, _ = nat.evaluate(_corpus(args), args.systems, args.split, acoustic=args.acoustic, ci=not args.no_ci)
    if report.empty:
        sys.exit("Nothing to evaluate.")
    print(report.round(3).T.to_string(header=False))


def cmd_listening_kit(args):
    trials = nat.build_listening_test(_corpus(args), args.systems, args.split, args.seed)
    print(f"{len(trials)} trials ({trials.system.value_counts().to_dict()}) in {args.root}/listening")


def cmd_listening_report(args):
    report, per_listener = nat.listening_report(_corpus(args))
    if report.empty:
        sys.exit("No responses yet.")
    print(report.round(3).to_string(index=False))
    print("\nPer listener:\n" + per_listener.round(2).to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(nat.DEFAULT_ROOT), help="corpus folder (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create the corpus from a text file, one stimulus sentence per line")
    p.add_argument("--stimuli", required=True)
    p.add_argument("--speakers", nargs="+", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("split", help="fix train/dev/test by sentence")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--dev", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--test-speakers", nargs="*", default=[], help="speakers held out entirely (test only)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_split)

    sub.add_parser("check", help="recordings, splits and cloning-leakage check").set_defaults(fn=cmd_check)
    sub.add_parser("convert-audio", help="convert m4a/mp3 recordings to WAV (originals kept)").set_defaults(
        fn=cmd_convert_audio)

    p = sub.add_parser("generate", help="synthesize all stimuli with a TTS system")
    p.add_argument("--system", required=True, choices=[*nat.STOCK_SYSTEMS, *nat.CLONE_SYSTEMS])
    p.add_argument("--splits", nargs="+", default=list(nat.SPLITS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--speaker-consent",
        action="store_true",
        help="confirm explicit permission from every cloned speaker (required for clone-* systems)",
    )
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("align", help="MFA word alignment (one batch call, same settings for every source)")
    p.add_argument("--sources", nargs="+", required=True, help="human, clone_ref, or system names")
    p.add_argument("--acoustic", default=nat.DEFAULT_ACOUSTIC)
    p.add_argument("--dictionary", default=nat.DEFAULT_DICTIONARY, help="only used to read the phone set")
    p.add_argument(
        "--word-lexicon",
        help="optional UTF-8 word<TAB>phone phone lexicon for English/code-mixed words absent from the Hindi model",
    )
    p.add_argument("--mfa", default="mfa", help="MFA executable")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_align)

    p = sub.add_parser("import-alignments",
                       help="copy externally made TextGrids (validated against the stimuli) for one source")
    p.add_argument("--source", required=True, help="human or a system name")
    p.add_argument("--folder", required=True, help="<folder>/<speaker>/<sentence_id>.TextGrid")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_import_alignments)

    p = sub.add_parser("prompt", help="cut a 6-12 s cloning prompt from the aligned passage")
    p.add_argument("--speaker", required=True)
    p.set_defaults(fn=cmd_prompt)

    sub.add_parser("durations", help="word durations + human-machine diff (durations.csv, pairs.csv)").set_defaults(
        fn=cmd_durations)

    p = sub.add_parser("train", help="fit the duration-correction layer on train, select on dev")
    p.add_argument("--system", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser(
        "mms-finetune",
        help="fine-tune only the MMS-TTS Hindi stochastic duration predictor from MFA pairs",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=cmd_mms_finetune)

    p = sub.add_parser("correct", help="apply the layer, writing generated/<system>+dur")
    p.add_argument("--system", required=True)
    p.add_argument("--splits", nargs="+", default=["dev", "test"])
    p.add_argument("--oracle", action="store_true", help="use real human durations (ceiling, not a result)")
    p.set_defaults(fn=cmd_correct)

    p = sub.add_parser("evaluate", help="objective metrics per system on one split")
    p.add_argument("--systems", nargs="+", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--acoustic", action="store_true", help="also F0, speaker similarity, UTMOS, detector")
    p.add_argument("--no-ci", action="store_true")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("listening-kit", help="build the human-vs-machine listening test")
    p.add_argument("--systems", nargs="+", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(fn=cmd_listening_kit)

    sub.add_parser("listening-report", help="analyse listening-test responses").set_defaults(fn=cmd_listening_report)

    args = parser.parse_args(argv)
    try:
        args.fn(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
