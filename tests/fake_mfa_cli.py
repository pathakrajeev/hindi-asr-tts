#!/usr/bin/env python3
"""Stand-in for `mfa align CORPUS LEXICON ACOUSTIC OUT ...` used by tests/test_pipeline.py.

Finds the non-silent regions of each wav (one per word in the synthetic test speech) and writes a
long-format TextGrid with a words tier and a phones tier. Not an aligner: test plumbing only.
"""
import sys
from pathlib import Path

import librosa
import numpy as np


def textgrid(words, xmax):
    def tier(name, items):
        lines = [f'        class = "IntervalTier"', f'        name = "{name}"', "        xmin = 0",
                 f"        xmax = {xmax}", f"        intervals: size = {len(items)}"]
        for i, (a, b, label) in enumerate(items, 1):
            lines += [f"        intervals [{i}]:", f"            xmin = {a}", f"            xmax = {b}",
                      f'            text = "{label}"']
        return lines

    filled, t = [], 0.0
    for a, b, w in words:
        if a > t:
            filled.append((t, a, ""))
        filled.append((a, b, w))
        t = b
    if t < xmax:
        filled.append((t, xmax, ""))
    phones = [(a, b, "x" if w else "") for a, b, w in filled]
    out = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {xmax}",
           "tiers? <exists>", "size = 2", "item []:", "    item [1]:", *tier("words", filled),
           "    item [2]:", *tier("phones", phones)]
    return "\n".join(out) + "\n"


def main(argv):
    assert argv[0] == "align", argv
    corpus, out = Path(argv[1]), Path(argv[4])
    for wav in corpus.rglob("*.wav"):
        words = wav.with_suffix(".lab").read_text(encoding="utf-8").split()
        y, sr = librosa.load(wav, sr=None)
        regions = librosa.effects.split(y, top_db=30, frame_length=256, hop_length=64)
        if len(regions) == len(words):
            spans = [(a / sr, b / sr) for a, b in regions]
        else:  # even split as a fallback
            edges = np.linspace(0, len(y) / sr, len(words) + 1)
            spans = list(zip(edges[:-1], edges[1:]))
        target = out / wav.parent.name / f"{wav.stem}.TextGrid"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textgrid([(round(a, 4), round(b, 4), w) for (a, b), w in zip(spans, words)],
                                   round(len(y) / sr, 4)), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
