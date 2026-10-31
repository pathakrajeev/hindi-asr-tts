"""Words exactly at MFA's 30 ms floor must not be flagged 'short' by float rounding.

    python tests/test_flags.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import naturalness as nat  # noqa: E402
from fake_mfa_cli import textgrid  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    corpus = nat.init_corpus(Path(tmp) / "c", ["राम घर गया।", "सीता को देखा।", "घर है बड़ा।"], ["S01"])
    nat.make_splits(corpus)
    flagged = 0
    for start in np.round(np.arange(0.50, 1.50, 0.01), 2):  # 100 placements of a 30 ms middle word
        words = [(0.20, start, "राम"), (start, round(start + 0.03, 2), "घर"), (round(start + 0.03, 2), 2.0, "गया")]
        grid = corpus.textgrid(nat.HUMAN, "S01", "sent_001")
        grid.parent.mkdir(parents=True, exist_ok=True)
        grid.write_text(textgrid(words, 2.2), encoding="utf-8")
        sf.write(corpus.root / "speakers/S01/stimuli/sent_001.wav", np.zeros(int(2.2 * 16000), np.float32), 16000)
        d = nat.build_durations(corpus)
        flagged += int((d.flag == "short").sum())
    print(("PASS" if flagged == 0 else "FAIL") + f" 30 ms words flagged short: {flagged} of 100")
    sys.exit(1 if flagged else 0)
