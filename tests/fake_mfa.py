"""Stand-in for `mfa align` in tests: finds word bursts by energy and writes MFA-style TextGrids.

Test audio renders each word as a Hann-enveloped tone, so word edges show as energy dips even when
words abut. Usage mirrors MFA: fake_mfa.py align CORPUS LEXICON ACOUSTIC OUTPUT [options].
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def segments(y, sr, threshold=0.03, min_run_s=0.025):
    hop, win = int(0.002 * sr), int(0.008 * sr)
    squared = np.square(y, dtype=np.float64)
    padded = np.pad(squared, (win // 2, win - 1 - win // 2))
    cumulative = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    power = (cumulative[win:] - cumulative[:-win]) / win
    env = np.sqrt(power[::hop])
    on = env > threshold * env.max()
    runs, start = [], None
    for i, flag in enumerate(np.append(on, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start * hop / sr, i * hop / sr))
            start = None
    return [(a, b) for a, b in runs if b - a >= min_run_s]


def textgrid(path, duration, words, lexicon):
    def tier(name, items):
        lines = [f'        class = "IntervalTier"', f'        name = "{name}"', "        xmin = 0",
                 f"        xmax = {duration}", f"        intervals: size = {len(items)}"]
        for k, (s, e, label) in enumerate(items, 1):
            lines += [f"        intervals [{k}]:", f"            xmin = {s}", f"            xmax = {e}",
                      f'            text = "{label}"']
        return lines

    word_items, phone_items, t = [], [], 0.0
    for (s, e), w in words:
        if s > t:
            word_items.append((t, s, "")); phone_items.append((t, s, "sil"))
        word_items.append((s, e, w))
        phones = lexicon.get(w, ["spn"])
        edges = np.linspace(s, e, len(phones) + 1)
        phone_items += [(edges[i], edges[i + 1], p) for i, p in enumerate(phones)]
        t = e
    if t < duration:
        word_items.append((t, duration, "")); phone_items.append((t, duration, "sil"))
    body = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {duration}",
            "tiers? <exists>", "size = 2", "item []:", "    item [1]:", *tier("words", word_items),
            "    item [2]:", *tier("phones", phone_items)]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def main():
    _, corpus, lexicon_path, _acoustic, out = sys.argv[1:6]
    lexicon = {}
    for line in Path(lexicon_path).read_text(encoding="utf-8").splitlines():
        word, *phones = line.split()
        lexicon[word] = phones
    for wav in Path(corpus).rglob("*.wav"):
        words = wav.with_suffix(".lab").read_text(encoding="utf-8").split()
        y, sr = sf.read(wav)
        runs = segments(y, sr)
        if len(runs) != len(words):  # MFA leaves unalignable files out
            print(f"cannot align {wav.name}: {len(runs)} bursts for {len(words)} words", file=sys.stderr)
            continue
        target = Path(out) / wav.parent.name / f"{wav.stem}.TextGrid"
        target.parent.mkdir(parents=True, exist_ok=True)
        textgrid(target, len(y) / sr, list(zip(runs, words)), lexicon)


if __name__ == "__main__":
    main()
