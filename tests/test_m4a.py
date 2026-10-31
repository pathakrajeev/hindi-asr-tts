"""Phone recordings (m4a) must not crash the corpus check, and must convert cleanly to WAV.

    python tests/test_m4a.py        (needs ffmpeg on PATH)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import naturalness as nat  # noqa: E402


class Upload:
    def __init__(self, path: Path, name: str):
        self.name, self._raw = name, path.read_bytes()

    def getvalue(self):
        return self._raw


def m4a(path: Path, seconds=1.5, sr=44_100):
    wav = path.with_suffix(".tmp.wav")
    t = np.arange(int(seconds * sr)) / sr
    sf.write(wav, (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "aac", str(path)], check=True)
    wav.unlink()


def check(ok, message):
    print(("PASS " if ok else "FAIL ") + message)
    if not ok:
        raise SystemExit(1)


def main():
    if not shutil.which("ffmpeg"):
        print("SKIP: ffmpeg not found")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = nat.init_corpus(tmp / "c", ["पहला वाक्य यहाँ है।", "दूसरा वाक्य यहाँ है।", "तीसरा वाक्य यहाँ है।"], ["S01"])
        nat.make_splits(corpus)

        # The reported situation: an m4a already sitting in stimuli/
        m4a(corpus.root / "speakers" / "S01" / "stimuli" / "sent_001.m4a")
        issues = nat.validate_corpus(corpus)  # used to raise LibsndfileError
        check(any("not WAV" in i["message"] for i in issues), "corpus check runs on m4a and warns")
        nat.build_durations(corpus)  # no TextGrids yet; must not crash either

        result = nat.convert_recordings_to_wav(corpus, log=lambda *_: None)
        wav = corpus.root / "speakers" / "S01" / "stimuli" / "sent_001.wav"
        check(result["converted"] == 1 and wav.is_file(), "existing m4a converted to WAV")
        check(abs(sf.info(str(wav)).duration - 1.5) < 0.1, "duration preserved")
        check((corpus.root / "speakers" / "S01" / "_original_audio" / "stimuli" / "sent_001.m4a").is_file(),
              "original kept in _original_audio")
        check(not nat.non_wav_recordings(corpus), "no non-WAV recordings left")

        # New uploads in m4a are stored as WAV
        src = tmp / "up.m4a"
        m4a(src)
        counts = nat.import_recordings(corpus, [Upload(src, "S01__sent_002.m4a")])
        stored = corpus.audio(nat.HUMAN, "S01", "sent_002")
        check(counts["stimuli"] == 1 and stored is not None and stored.suffix == ".wav", "m4a upload stored as WAV")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
