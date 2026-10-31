"""Human-likeness pipeline: MFA word durations, duration-error feedback, voice cloning, evaluation.

Corpus layout (root = NATURALNESS_ROOT, default ./data/naturalness):

    stimuli.csv                              sentence_id,text  (the experimental stimuli)
    splits.csv                               sentence_id,split (train/dev/test, fixed before analysis)
    config.json                              {"test_speakers": [...]} speakers held out entirely
    speakers/<spk>/clone_ref/passage.wav     read-aloud passage used ONLY for voice cloning
    speakers/<spk>/clone_ref/passage.txt
    speakers/<spk>/clone_ref/prompt.wav|txt  6-12 s cloning prompt cut from the passage
    speakers/<spk>/stimuli/<sentence_id>.wav human recordings of the stimuli
    generated/<system>/<spk>/<sentence_id>.wav
    alignments/<source>/<spk>/<sentence_id>.TextGrid   source = "human", "clone_ref" or a system
    durations.csv                            one row per aligned word per source
    pairs.csv                                human vs machine word pairs with the duration diff
    models/<system>.joblib                   the duration-correction layer
    eval/                                    reports
    listening/                               human/machine discrimination test

The duration-correction layer learns log(human word duration / machine word duration) and the
human pause after each word from text and machine-alignment features, on the train split only.
It is applied to new machine speech by pitch-preserving per-word time-stretching (Praat
overlap-add) and pause resizing. Stock TTS APIs (Bulbul) expose no trainable duration predictor,
so the error is fed back through this layer rather than into the TTS weights.
"""
from __future__ import annotations

import difflib
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(os.getenv("NATURALNESS_ROOT", PROJECT_DIR / "data" / "naturalness"))
DEFAULT_DICTIONARY = os.getenv("INDICMFA_HI_DICTIONARY", str(PROJECT_DIR / "Hindi_Dict_g2g.txt"))
DEFAULT_ACOUSTIC = os.getenv("INDICMFA_HI_ACOUSTIC", str(PROJECT_DIR / "Hindi_All_Acoustic.zip"))

HUMAN = "human"
CLONE_REF = "clone_ref"
SPLITS = ("train", "dev", "test")
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
MFA_SR = 16_000
PAIR_COLUMNS = [
    "system", "speaker", "sentence_id", "word_idx", "split", "n_words", "punct_after", "word",
    "start_s_h", "end_s_h", "dur_s_h", "pause_after_s_h", "n_phones_h",
    "start_s_m", "end_s_m", "dur_s_m", "pause_after_s_m", "n_phones_m",
    "diff_s", "log_ratio", "log_ratio_resid", "pause_diff_s",
]

MIN_WORD_S = 0.03
MAX_WORD_S = 1.5
PAUSE_MIN_S = 0.10        # a gap at least this long counts as a pause
MAX_PAUSE_S = 1.2
STRETCH_LIMITS = (0.6, 1.6)
FADE_S = 0.005
LEAK_NGRAM = 4

PUNCT_CLASSES = {"comma": ",;:-–—", "question": "?", "final": "।॥.!"}
_ALL_PUNCT = "।॥!?,.;:\"'`()[]{}-–—_/\\|<>*&^%$#@+=~"
_PUNCT = re.compile("[" + re.escape(_ALL_PUNCT) + "]")
_DIGITS = re.compile(r"\d")

# Function words copied from speech_core so this module imports without torch/streamlit.
FUNCTION_WORDS = set("""
है हैं था थी थे हो होगा होगी का की के में से को और ने पर भी तो कि जो यह वह ये वो नहीं ही तक लिए कर रहा रही रहे
एक हम मैं आप तुम उस इस उन इन कुछ बहुत क्या कब कहाँ क्यों कैसे अब फिर गया गई गए जा जाता जाती
""".split())


# ---------------------------------------------------------------- text


def mfa_text(text: str) -> str:
    """The exact word sequence aligned for every source. Same rule as speech_core.normalize_reference_text."""
    normalized = unicodedata.normalize("NFC", str(text)).casefold()
    normalized = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in normalized)
    return " ".join(_PUNCT.sub(" ", normalized).split())


def tokens_with_punct(text: str) -> list[tuple[str, str]]:
    """(normalized word, punctuation class after it) for each aligned word.
    Punctuation-only tokens (e.g. a spaced danda) attach to the previous word."""
    out: list[list[str]] = []
    for raw in str(text).split():
        words = mfa_text(raw).split()
        trailing = raw[len(raw.rstrip(_ALL_PUNCT)):]
        cls = next((name for name, chars in PUNCT_CLASSES.items() if any(c in chars for c in trailing)), "none")
        if not words:
            if out and cls != "none":
                out[-1][1] = cls
            continue
        for word in words[:-1]:  # "a-b" splits into two words; punctuation follows the last
            out.append([word, "none"])
        out.append([words[-1], cls])
    return [(w, c) for w, c in out]


# Prosodic words. Hindi postpositions, auxiliaries/copula and particles are enclitics: spoken as one
# prosodic unit with the word before them. Forced aligners place the boundary inside such a pair
# unreliably (on pilot data MFA squeezed 40 of 57 bad words, mostly these, to 10-20 ms), while the
# pair's outer edges are reliable. So durations are measured on units = a word plus any enclitics that
# follow it without punctuation in between. The grouping comes from the TEXT alone, so it is identical
# for the human and every TTS system, and available for new sentences at correction time.
ENCLITICS = frozenset("का की के को से में पर ने तक है हैं था थी थे थीं हो हूँ हूं भी ही सा सी".split())
LONG_PER_EXTRA_WORD_S = 0.5   # a unit of k words may last MAX_WORD_S + 0.5 s x (k - 1)
COLLAPSE_MS_PER_PHONE = 25.0  # under this per acoustic unit (akshara), the aligner squeezed the word


def unit_spans(text: str) -> list[list[int]]:
    """Token indices of each prosodic unit (see ENCLITICS)."""
    tokens = tokens_with_punct(text)
    spans: list[list[int]] = []
    for i, (word, _) in enumerate(tokens):
        if spans and word in ENCLITICS and tokens[i - 1][1] == "none":
            spans[-1].append(i)
        else:
            spans.append([i])
    return spans


def unit_tokens(text: str) -> list[tuple[str, str]]:
    """(unit text, punctuation class after it), e.g. ("पानी की", "none")."""
    tokens = tokens_with_punct(text)
    return [(" ".join(tokens[i][0] for i in span), tokens[span[-1]][1]) for span in unit_spans(text)]


_CONSONANT = re.compile("[क-हक़-य़]")
_IND_VOWEL = re.compile("[ऄ-औॠॡ]")
_MATRA = re.compile("[ा-ौॢॣ]")
_VIRAMA = "्"


def word_shape(word: str) -> dict:
    deva = sum(1 for c in word if "ऀ" <= c <= "ॿ")
    if deva == 0:
        letters = sum(c.isalpha() for c in word)
        vowel_groups = len(re.findall(r"[aeiouy]+", word.lower()))
        return {"n_letters": letters, "n_syll": max(vowel_groups, 1), "n_matra": 0, "n_conjunct": 0, "is_latin": 1}
    consonants = len(_CONSONANT.findall(word))
    conjuncts = word.count(_VIRAMA)
    return {
        "n_letters": consonants + len(_IND_VOWEL.findall(word)),
        "n_syll": max(consonants - conjuncts + len(_IND_VOWEL.findall(word)), 1),
        "n_matra": len(_MATRA.findall(word)),
        "n_conjunct": conjuncts,
        "is_latin": 0,
    }


def leakage_report(passage_text: str, stimuli: pd.DataFrame, n: int = LEAK_NGRAM) -> pd.DataFrame:
    """Longest contiguous word run each stimulus shares with the cloning passage.
    A stimulus leaks if it shares n words in a row (or is entirely contained, when shorter)."""
    passage = mfa_text(passage_text).split()
    rows = []
    for row in stimuli.itertuples():
        sentence = mfa_text(row.text).split()
        match = difflib.SequenceMatcher(None, passage, sentence, autojunk=False).find_longest_match(
            0, len(passage), 0, len(sentence))
        longest = match.size
        rows.append({"sentence_id": row.sentence_id, "words": len(sentence), "longest_shared_run": longest,
                     "shared_text": " ".join(sentence[match.b: match.b + longest]),
                     "leak": bool(sentence) and longest >= min(n, len(sentence))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- corpus


def _safe(name: str) -> str:
    return re.sub(r"[^\w+.-]", "_", name)


def _validate_id(value: str, label: str) -> str:
    value = str(value).strip()
    if not value or value in {".", ".."} or not re.fullmatch(r"[\w.-]+", value):
        raise ValueError(f"{label} must contain only letters, numbers, underscores, dots, or hyphens.")
    return value


@dataclass
class Corpus:
    root: Path = DEFAULT_ROOT

    def __post_init__(self):
        self.root = Path(self.root)

    # files
    @property
    def stimuli_csv(self): return self.root / "stimuli.csv"
    @property
    def splits_csv(self): return self.root / "splits.csv"
    @property
    def config_json(self): return self.root / "config.json"
    @property
    def durations_csv(self): return self.root / "durations.csv"
    @property
    def pairs_csv(self): return self.root / "pairs.csv"

    def config(self) -> dict:
        return json.loads(self.config_json.read_text(encoding="utf-8")) if self.config_json.is_file() else {}

    def stimuli(self) -> pd.DataFrame:
        if not self.stimuli_csv.is_file():
            raise FileNotFoundError(f"{self.stimuli_csv} not found. Run `init` first.")
        return pd.read_csv(self.stimuli_csv, dtype=str, keep_default_na=False)

    def text_of(self) -> dict[str, str]:
        frame = self.stimuli()
        return dict(zip(frame.sentence_id, frame.text))

    def splits(self) -> dict[str, str]:
        if not self.splits_csv.is_file():
            raise FileNotFoundError(f"{self.splits_csv} not found. Run `split` first.")
        frame = pd.read_csv(self.splits_csv, dtype=str, keep_default_na=False)
        return dict(zip(frame.sentence_id, frame.split))

    def speakers(self) -> list[str]:
        folder = self.root / "speakers"
        return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []

    def systems(self) -> list[str]:
        folder = self.root / "generated"
        return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []

    def clone_dir(self, speaker: str) -> Path:
        return self.root / "speakers" / speaker / "clone_ref"

    def audio(self, source: str, speaker: str, sentence_id: str) -> Path | None:
        if source == HUMAN:
            folder, stem = self.root / "speakers" / speaker / "stimuli", sentence_id
        elif source == CLONE_REF:
            folder, stem = self.clone_dir(speaker), "passage"
        else:
            folder, stem = self.root / "generated" / source / speaker, sentence_id
        return next((folder / f"{stem}{s}" for s in AUDIO_SUFFIXES if (folder / f"{stem}{s}").is_file()), None)

    def generated_path(self, system: str, speaker: str, sentence_id: str) -> Path:
        return self.root / "generated" / system / speaker / f"{sentence_id}.wav"

    def textgrid(self, source: str, speaker: str, sentence_id: str) -> Path:
        return self.root / "alignments" / source / speaker / f"{sentence_id}.TextGrid"

    def model_path(self, system: str) -> Path:
        return self.root / "models" / f"{_safe(system)}.joblib"

    @property
    def mms_finetune_path(self) -> Path:
        return self.root / "models" / "mms-tts-hin-duration.pt"

    def split_of(self, speaker: str, sentence_id: str, splits: dict | None = None) -> str:
        """Held-out speakers contribute only test sentences, so they are unseen in voice and text."""
        split = (splits or self.splits()).get(sentence_id, "unassigned")
        if speaker in self.config().get("test_speakers", []):
            return "test" if split == "test" else "unused"
        return split


README = """# Naturalness corpus

1. `stimuli.csv` lists the experimental sentences. Spell numbers out as words.
2. `splits.csv` fixes train/dev/test BY SENTENCE. Do not edit after you start analysing.
3. For each speaker, `speakers/<spk>/`:
   - `clone_ref/passage.wav` + `passage.txt`: 2-5 minutes of read-aloud text that shares NO
     sentence with the stimuli. Used only for voice cloning (check with `check`).
   - `stimuli/<sentence_id>.wav`: one recording per stimulus sentence. Same mic and room.
4. Run the CLI steps in NATURALNESS.md.
"""


def init_corpus(root: Path, sentences: list[str], speakers: list[str], force: bool = False) -> Corpus:
    corpus = Corpus(root)
    if corpus.stimuli_csv.exists() and not force:
        raise FileExistsError(f"{corpus.stimuli_csv} exists; pass force to overwrite.")
    if corpus.stimuli_csv.exists() and force:
        populated = (
            corpus.splits_csv.exists()
            or corpus.durations_csv.exists()
            or corpus.pairs_csv.exists()
            or (corpus.root / "generated").exists()
            or (corpus.root / "alignments").exists()
        )
        if populated:
            raise ValueError(
                "This corpus already has analysis data. Create a new corpus folder rather than replacing "
                "stimuli and invalidating existing splits or alignments."
            )
    if not speakers:
        raise ValueError("At least one speaker ID is required.")
    speakers = [_validate_id(speaker, "Speaker ID") for speaker in speakers]
    if len(set(speakers)) != len(speakers):
        raise ValueError("Speaker IDs must be unique.")
    sentences = [" ".join(s.split()) for s in sentences if s.strip()]
    if not sentences:
        raise ValueError("At least one non-empty stimulus sentence is required.")
    if len(set(map(mfa_text, sentences))) != len(sentences):
        raise ValueError("Duplicate stimulus sentences found; each sentence must be unique.")
    width = max(3, len(str(len(sentences))))
    corpus.root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sentence_id": [f"sent_{i:0{width}d}" for i in range(1, len(sentences) + 1)],
                  "text": sentences}).to_csv(corpus.stimuli_csv, index=False, encoding="utf-8")
    for speaker in speakers:
        (corpus.root / "speakers" / speaker / "stimuli").mkdir(parents=True, exist_ok=True)
        corpus.clone_dir(speaker).mkdir(parents=True, exist_ok=True)
    (corpus.root / "README.md").write_text(README, encoding="utf-8")
    return corpus


def append_stimuli(corpus: Corpus, sentences: list[str]) -> pd.DataFrame:
    """Append stimuli to an incomplete corpus before recordings or analysis begin."""
    if corpus.splits_csv.exists():
        raise FileExistsError("Stimuli are frozen after splits.csv is created.")
    if not corpus.stimuli_csv.is_file():
        raise FileNotFoundError(f"{corpus.stimuli_csv} not found. Create the corpus first.")
    protected = (
        corpus.durations_csv,
        corpus.pairs_csv,
        corpus.root / "generated",
        corpus.root / "alignments",
        corpus.root / "models",
        corpus.root / "eval",
    )
    if any(path.is_file() if path.suffix else path.is_dir() for path in protected):
        raise ValueError("This corpus already contains analysis data; do not change its stimuli.")
    speaker_audio = corpus.root / "speakers"
    if speaker_audio.is_dir() and any(
        path.is_file()
        for speaker_dir in speaker_audio.iterdir()
        for path in (speaker_dir / "stimuli").glob("*")
    ):
        raise ValueError("Human stimulus recordings already exist; do not change the corpus stimuli.")

    frame = corpus.stimuli()
    additions = [" ".join(sentence.split()) for sentence in sentences if sentence.strip()]
    if not additions:
        raise ValueError("Enter at least one additional non-empty stimulus sentence.")
    existing_normalized = set(frame.text.map(mfa_text))
    added_normalized = [mfa_text(sentence) for sentence in additions]
    if len(set(added_normalized)) != len(added_normalized):
        raise ValueError("The additional stimulus list contains duplicates.")
    duplicates = existing_normalized.intersection(added_normalized)
    if duplicates:
        raise ValueError("An additional stimulus duplicates text already in the corpus.")

    used_ids = set(frame.sentence_id)
    next_id = len(frame) + 1
    width = max(3, len(str(next_id + len(additions) - 1)))
    new_rows = []
    for sentence in additions:
        while f"sent_{next_id:0{width}d}" in used_ids:
            next_id += 1
        sentence_id = f"sent_{next_id:0{width}d}"
        used_ids.add(sentence_id)
        new_rows.append({"sentence_id": sentence_id, "text": sentence})
        next_id += 1

    updated = pd.concat([frame, pd.DataFrame(new_rows)], ignore_index=True)
    temp_path = corpus.stimuli_csv.with_suffix(".csv.tmp")
    try:
        updated.to_csv(temp_path, index=False, encoding="utf-8")
        temp_path.replace(corpus.stimuli_csv)
    finally:
        temp_path.unlink(missing_ok=True)
    return updated


def import_recordings(corpus: Corpus, audio_uploads, transcript_uploads=()) -> dict[str, int]:
    """Import `speaker__sentence_id.wav` and separate `speaker__passage.wav/.txt` files."""
    if not corpus.stimuli_csv.is_file():
        raise FileNotFoundError(f"{corpus.stimuli_csv} not found. Create the corpus first.")
    known_speakers, known_sentences = set(corpus.speakers()), set(corpus.text_of())
    audio_items, transcript_items, transcript_keys = {}, {}, set()
    verified_stimulus_transcripts = 0

    def parse_upload(upload, suffixes, label):
        name = Path(upload.name).name
        suffix = Path(name).suffix.lower()
        if suffix not in suffixes:
            raise ValueError(f"{name}: unsupported {label} file type.")
        stem = Path(name).stem
        if "__" not in stem:
            raise ValueError(f"{name}: name it speaker__sentence_id or speaker__passage.")
        speaker, item_id = stem.split("__", maxsplit=1)
        speaker = _validate_id(speaker, "Speaker ID")
        if speaker not in known_speakers:
            raise ValueError(f"{name}: unknown speaker ID {speaker!r}.")
        return name, speaker, item_id, suffix

    for upload in audio_uploads:
        name, speaker, item_id, suffix = parse_upload(upload, AUDIO_SUFFIXES, "audio")
        is_passage = item_id == "passage"
        if not is_passage and item_id not in known_sentences:
            raise ValueError(f"{name}: unknown stimulus ID {item_id!r}.")
        key = (speaker, item_id)
        if key in audio_items:
            raise ValueError(f"More than one audio file was uploaded for {speaker}/{item_id}.")
        raw = upload.getvalue()
        if not raw:
            raise ValueError(f"{name}: file is empty.")
        if len(raw) > 50 * 1024 * 1024:
            raise ValueError(f"{name}: audio exceeds the 50 MB limit.")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(raw)
            temp_path = Path(temp.name)
        try:
            audio, sr = _load_mono(temp_path)
            if not len(audio) or not np.isfinite(audio).all() or sr <= 0:
                raise ValueError(f"{name}: audio is empty or invalid.")
        except Exception as exc:
            raise ValueError(f"{name}: could not decode audio ({type(exc).__name__}: {exc}).") from exc
        finally:
            temp_path.unlink(missing_ok=True)
        if suffix != ".wav":
            # Phone formats (m4a, mp3) are stored as 16-bit WAV at their own sample rate.
            buffer = io.BytesIO()
            sf.write(buffer, audio, sr, format="WAV", subtype="PCM_16")
            raw, suffix = buffer.getvalue(), ".wav"
        if is_passage:
            target = corpus.clone_dir(speaker) / f"passage{suffix}"
        else:
            target = corpus.root / "speakers" / speaker / "stimuli" / f"{item_id}{suffix}"
        if corpus.audio(CLONE_REF if is_passage else HUMAN, speaker, item_id) is not None:
            raise FileExistsError(f"A recording already exists for {speaker}/{item_id}; remove it manually before replacing.")
        audio_items[key] = (target, raw, is_passage)

    for upload in transcript_uploads:
        name, speaker, item_id, _ = parse_upload(upload, (".txt",), "transcript")
        if item_id != "passage" and item_id not in known_sentences:
            raise ValueError(f"{name}: unknown stimulus ID {item_id!r}.")
        key = (speaker, item_id)
        if key in transcript_keys:
            raise ValueError(f"More than one transcript was uploaded for {speaker}/{item_id}.")
        transcript_keys.add(key)
        try:
            text = upload.getvalue().decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name}: transcript must be UTF-8 text.") from exc
        if not text:
            raise ValueError(f"{name}: transcript is empty.")
        if item_id == "passage":
            transcript_items[key] = (corpus.clone_dir(speaker) / "passage.txt", text)
        elif mfa_text(text) != mfa_text(corpus.text_of()[item_id]):
            raise ValueError(
                f"{name}: transcript does not match the frozen stimulus text for {item_id}. "
                "Correct the transcript or remove this optional verification file."
            )
        else:
            verified_stimulus_transcripts += 1

    for key, (_, _, is_passage) in audio_items.items():
        if is_passage and key not in transcript_items:
            raise ValueError(f"{key[0]}__passage.wav requires a matching {key[0]}__passage.txt transcript.")
    for key in transcript_items:
        if key not in audio_items and not (corpus.clone_dir(key[0]) / "passage.wav").is_file():
            raise ValueError(f"{key[0]}__passage.txt requires a matching read-aloud audio file.")
    for target, _, _ in audio_items.values():
        if target.exists():
            raise FileExistsError(f"{target} already exists; existing recordings are never overwritten.")
    for target, _ in transcript_items.values():
        if target.exists():
            raise FileExistsError(f"{target} already exists; existing transcripts are never overwritten.")

    for target, raw, _ in audio_items.values():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    for target, text in transcript_items.values():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return {
        "stimuli": sum(not is_passage for _, _, is_passage in audio_items.values()),
        "passages": sum(is_passage for _, _, is_passage in audio_items.values()),
        "verified_transcripts": verified_stimulus_transcripts,
    }


def make_splits(corpus: Corpus, train: float = 0.70, dev: float = 0.15, seed: int = 13,
                test_speakers: list[str] | None = None, force: bool = False) -> pd.DataFrame:
    if corpus.splits_csv.exists() and not force:
        raise FileExistsError("splits.csv already exists. Splits must stay fixed; pass force only before any analysis.")
    ids = corpus.stimuli().sentence_id.tolist()
    if len(ids) < 3:
        raise ValueError("At least three unique stimuli are required to create non-empty train, dev, and test splits.")
    if not 0 < train < 1 or not 0 < dev < 1 or train + dev >= 1:
        raise ValueError("Train and dev fractions must be positive and sum to less than 1.")
    unknown_speakers = set(test_speakers or []) - set(corpus.speakers())
    if unknown_speakers:
        raise ValueError(f"Unknown held-out speaker(s): {', '.join(sorted(unknown_speakers))}")
    order = np.random.default_rng(seed).permutation(len(ids))
    n_train, n_dev = round(train * len(ids)), round(dev * len(ids))
    n_train = min(max(n_train, 1), len(ids) - 2)
    n_dev = min(max(n_dev, 1), len(ids) - n_train - 1)
    split = np.empty(len(ids), dtype=object)
    split[order[:n_train]], split[order[n_train:n_train + n_dev]], split[order[n_train + n_dev:]] = "train", "dev", "test"
    frame = pd.DataFrame({"sentence_id": ids, "split": split})
    frame.to_csv(corpus.splits_csv, index=False)
    corpus.config_json.write_text(json.dumps({"seed": seed, "test_speakers": test_speakers or []}, indent=2))
    return frame


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_corpus(corpus: Corpus) -> list[dict]:
    """Issues as {level: error|warning|ok, message}. Any 'error' blocks cloning and evaluation."""
    issues = []
    add = lambda level, message: issues.append({"level": level, "message": message})  # noqa: E731
    try:
        stimuli = corpus.stimuli()
    except FileNotFoundError as exc:
        return [{"level": "error", "message": str(exc)}]
    digits = stimuli[stimuli.text.str.contains(_DIGITS)]
    if len(digits):
        add("warning", f"{len(digits)} stimuli contain digits ({', '.join(digits.sentence_id[:5])}). "
                       "Spell numbers out so human and TTS read identical words.")
    try:
        splits = corpus.splits()
        counts = pd.Series(splits).value_counts().to_dict()
        add("ok", f"Splits: {counts}")
        if set(stimuli.sentence_id) - set(splits):
            add("error", "Some stimuli have no split. Re-run `split --force` before analysis begins.")
        if set(splits.values()) - set(SPLITS):
            add("error", f"Unknown split names: {sorted(set(splits.values()) - set(SPLITS))}")
        missing_splits = set(SPLITS) - set(splits.values())
        if missing_splits:
            add("error", f"Empty split(s): {', '.join(sorted(missing_splits))}. Recreate the split before analysis.")
    except FileNotFoundError as exc:
        add("error", str(exc))
    speakers = corpus.speakers()
    if not speakers:
        add("error", "No speakers in speakers/.")
    stim_hashes = {}
    not_wav = non_wav_recordings(corpus)
    if not_wav:
        add("warning", f"{len(not_wav)} recording(s) are not WAV (e.g. {not_wav[0].name}). They still work, but "
                       "converting them once to WAV makes every later step faster and more reliable "
                       "(Convert recordings to WAV / `convert-audio`).")
    for speaker in speakers:
        recorded = [sid for sid in stimuli.sentence_id if corpus.audio(HUMAN, speaker, sid)]
        add("ok" if len(recorded) == len(stimuli) else "warning",
            f"{speaker}: {len(recorded)}/{len(stimuli)} stimuli recorded")
        for sid in recorded:
            path = corpus.audio(HUMAN, speaker, sid)
            stim_hashes[_file_hash(path)] = f"{speaker}/{sid}"
            if _audio_info(path)[0] < MFA_SR:
                add("warning", f"{speaker}/{sid}: sample rate below 16 kHz")
        passage_audio = corpus.audio(CLONE_REF, speaker, "passage")
        passage_text = corpus.clone_dir(speaker) / "passage.txt"
        if not passage_audio or not passage_text.is_file():
            add("warning", f"{speaker}: no clone_ref/passage.wav + passage.txt (needed for voice cloning)")
            continue
        seconds = _audio_info(passage_audio)[1]
        if seconds < 60:
            add("warning", f"{speaker}: cloning passage is {seconds:.0f} s; record 2-5 minutes")
        leaks = leakage_report(passage_text.read_text(encoding="utf-8"), stimuli)
        bad = leaks[leaks.leak]
        if len(bad):
            add("error", f"{speaker}: cloning passage shares text with {len(bad)} stimuli "
                         f"({', '.join(bad.sentence_id[:5])}): this leaks test durations into the clone.")
        else:
            add("ok", f"{speaker}: cloning passage shares at most {leaks.longest_shared_run.max()} consecutive "
                      f"words with any stimulus (limit {LEAK_NGRAM - 1})")
        prompt_text = corpus.clone_dir(speaker) / "prompt.txt"
        if prompt_text.is_file():
            prompt_leaks = leakage_report(prompt_text.read_text(encoding="utf-8"), stimuli)
            if prompt_leaks.leak.any():
                add("error", f"{speaker}: clone_ref/prompt.txt shares text with stimuli "
                             f"({', '.join(prompt_leaks[prompt_leaks.leak].sentence_id[:5])}); re-cut it from the passage.")
        clone_files = [p for p in corpus.clone_dir(speaker).iterdir() if p.suffix.lower() in AUDIO_SUFFIXES]
        for path in clone_files:
            if _file_hash(path) in stim_hashes:
                add("error", f"{speaker}: {path.name} is identical to stimulus recording {stim_hashes[_file_hash(path)]}")
    return issues


def has_errors(issues: list[dict]) -> bool:
    return any(i["level"] == "error" for i in issues)


# ---------------------------------------------------------------- TextGrid


_TG_TOKEN = re.compile(r'"((?:[^"]|"")*)"|([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)|<exists>')


def read_textgrid(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    """Long or short Praat TextGrid -> {tier name: [(start, end, label)]}. Point tiers use start == end."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    text = re.sub(r"\[\s*\d+\s*\]", "", text)  # drop "item [1]:" / "intervals [3]:" indices
    tokens = [m.group(1).replace('""', '"') if m.group(1) is not None else (float(m.group(2)) if m.group(2) else None)
              for m in _TG_TOKEN.finditer(text)]
    tokens = [t for t in tokens if t is not None]
    pos = 2  # "ooTextFile", "TextGrid"
    pos += 2  # xmin xmax
    n_tiers = int(tokens[pos]); pos += 1
    tiers = {}
    for _ in range(n_tiers):
        kind, name = tokens[pos], tokens[pos + 1]
        pos += 4  # class, name, xmin, xmax
        n = int(tokens[pos]); pos += 1
        items = []
        for _ in range(n):
            if kind == "IntervalTier":
                items.append((float(tokens[pos]), float(tokens[pos + 1]), str(tokens[pos + 2]))); pos += 3
            else:
                items.append((float(tokens[pos]), float(tokens[pos]), str(tokens[pos + 1]))); pos += 2
        tiers[str(name)] = items
    return tiers


def _tier(tiers: dict, suffix: str):
    return next((v for k, v in tiers.items() if k.lower().strip().endswith(suffix)), None)


def word_intervals(path: Path, audio_seconds: float | None = None) -> list[dict]:
    """Words with start/end/duration, the gap to the next word, phone count and an OOV flag."""
    tiers = read_textgrid(path)
    words = _tier(tiers, "words") or _tier(tiers, "word")
    if words is None:
        raise ValueError(f"{path}: no words tier")
    phones = _tier(tiers, "phones") or []
    spoken = [(s, e, w) for s, e, w in words if w.strip() and w.strip() not in ("<eps>", "sil", "sp")]
    end_of_file = audio_seconds if audio_seconds is not None else (words[-1][1] if words else 0.0)
    rows = []
    for i, (start, end, word) in enumerate(spoken):
        inside = [p for s, e, p in phones if s >= start - 1e-4 and e <= end + 1e-4 and p.strip() and p not in ("sil", "sp")]
        next_start = spoken[i + 1][0] if i + 1 < len(spoken) else end_of_file
        oov = word.strip().casefold() in {"<unk>", "spn"} or not inside or all(
            p.strip().casefold() in {"spn", "<unk>"} for p in inside
        )
        rows.append({"word_idx": i, "word": word, "start_s": start, "end_s": end, "dur_s": end - start,
                     "pause_after_s": max(next_start - end, 0.0), "n_phones": len(inside), "oov": oov})
    return rows


def merge_units(words: list[dict], spans: list[list[int]]) -> list[dict]:
    """Word intervals -> prosodic-unit intervals (outer edges of each span)."""
    units = []
    for k, span in enumerate(spans):
        first, last = words[span[0]], words[span[-1]]
        units.append({"word_idx": k, "word": " ".join(words[i]["word"] for i in span),
                      "start_s": first["start_s"], "end_s": last["end_s"], "dur_s": last["end_s"] - first["start_s"],
                      "pause_after_s": last["pause_after_s"], "n_phones": sum(words[i]["n_phones"] for i in span),
                      "oov": any(words[i]["oov"] for i in span), "n_tokens": len(span)})
    return units


def sentence_units(grid: Path, text: str, audio_seconds: float | None = None) -> tuple[list[dict], bool]:
    """(prosodic-unit intervals, mismatch). On a word-sequence mismatch the raw words are returned."""
    words = word_intervals(grid, audio_seconds)
    if [w["word"] for w in words] != [w for w, _ in tokens_with_punct(text)]:
        return [{**w, "n_tokens": 1} for w in words], True
    return merge_units(words, unit_spans(text)), False


# ---------------------------------------------------------------- MFA


def phone_inventory(acoustic: str, dictionary: str | None = None) -> set[str]:
    """Phones of the acoustic model (IndicMFA uses akshara units such as का, क्ष, ्)."""
    path = Path(acoustic)
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            meta = next((n for n in archive.namelist() if n.endswith("meta.json")), None)
            if meta:
                phones = json.loads(archive.read(meta).decode("utf-8")).get("phones") or []
                if phones:
                    return {p for p in phones if p not in ("sil", "spn", "<eps>")}
    if dictionary and Path(dictionary).is_file():
        return {p for line in Path(dictionary).read_text(encoding="utf-8").splitlines() for p in line.split()[1:]}
    return set()


def pronounce(word: str, phones: set[str]) -> list[str]:
    """Greedy longest-match segmentation of a word into acoustic-model phones; unknown marks are dropped,
    nukta forms fall back to the plain consonant. Returns [] when nothing matches (an OOV word)."""
    word = unicodedata.normalize("NFD", word).replace("़", "")  # क़ -> क (nukta)
    word = unicodedata.normalize("NFC", word)
    longest = max((len(p) for p in phones), default=1)
    out, i = [], 0
    while i < len(word):
        for size in range(min(longest, len(word) - i), 0, -1):
            if word[i:i + size] in phones:
                out.append(word[i:i + size]); i += size
                break
        else:
            i += 1
    return out


def load_word_lexicon(path: str | Path | None, phones: set[str]) -> dict[str, list[str]]:
    """Load optional word-to-phone overrides; each line is `word<TAB>phone phone ...`."""
    if path is None or not str(path).strip():
        return {}
    lexicon_path = Path(path)
    if not lexicon_path.is_file():
        raise FileNotFoundError(f"Supplemental MFA word lexicon not found: {lexicon_path}")
    overrides = {}
    for line_number, raw in enumerate(lexicon_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t", maxsplit=1)
        if len(fields) != 2:
            fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"{lexicon_path}:{line_number}: expected a word followed by phone symbols.")
        token = mfa_text(fields[0]).split()
        pronunciation = fields[1].split()
        if len(token) != 1 or not pronunciation:
            raise ValueError(f"{lexicon_path}:{line_number}: expected exactly one word and at least one phone.")
        unknown = set(pronunciation) - phones
        if unknown:
            raise ValueError(
                f"{lexicon_path}:{line_number}: phone(s) absent from the acoustic model: "
                f"{', '.join(sorted(unknown))}"
            )
        if token[0] in overrides:
            raise ValueError(f"{lexicon_path}:{line_number}: duplicate pronunciation for {token[0]!r}.")
        overrides[token[0]] = pronunciation
    return overrides


def write_lexicon(words: set[str], phones: set[str], path: Path,
                  overrides: dict[str, list[str]] | None = None) -> list[str]:
    """Write MFA pronunciations, preferring explicit overrides; return unresolved words."""
    overrides = overrides or {}
    missing, lines = [], []
    for word in sorted(words):
        pron = overrides.get(word) or pronounce(word, phones)
        if pron:
            lines.append(f"{word}\t{' '.join(pron)}")
        else:
            missing.append(word)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return missing


def _load_mono(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    try:
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except sf.LibsndfileError:  # m4a/aac (and mp3 on older libsndfile): decode with ffmpeg
        audio, rate = _ffmpeg_decode(Path(path))
    if sr and rate != sr:
        audio, rate = librosa.resample(audio, orig_sr=rate, target_sr=sr), sr
    return audio.astype(np.float32), int(rate)


def _ffmpeg_decode(path: Path) -> tuple[np.ndarray, int]:
    """Decode any ffmpeg-readable file to mono float32 at its native rate (falls back to librosa)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            audio, rate = librosa.load(str(path), sr=None, mono=True)
            return audio.astype(np.float32), int(rate)
        except Exception as exc:
            raise RuntimeError(f"Cannot decode {path.name}: install ffmpeg, or convert it to WAV first.") from exc
    with tempfile.TemporaryDirectory(prefix="naturalness_decode_") as folder:
        out = Path(folder) / "decoded.wav"
        proc = subprocess.run([ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
                               "-c:a", "pcm_s16le", str(out)], capture_output=True, text=True, timeout=300)
        if proc.returncode or not out.is_file():
            raise RuntimeError(f"ffmpeg could not decode {path.name}: {proc.stderr.strip()[-300:]}")
        audio, rate = sf.read(str(out), dtype="float32")
    return np.asarray(audio, np.float32), int(rate)


def _audio_info(path: Path) -> tuple[int, float]:
    """(sample rate, seconds). soundfile reads WAV/FLAC/OGG headers; phone formats such as m4a/mp3
    are decoded instead (needs ffmpeg), so they never crash the corpus check."""
    try:
        info = sf.info(str(path))
        return int(info.samplerate), float(info.duration)
    except (sf.LibsndfileError, RuntimeError):
        audio, rate = _load_mono(Path(path))
        return rate, len(audio) / rate


def non_wav_recordings(corpus: "Corpus") -> list[Path]:
    """Human stimulus recordings and cloning passages stored in a format other than WAV."""
    out = []
    for speaker in corpus.speakers():
        for folder in (corpus.root / "speakers" / speaker / "stimuli", corpus.clone_dir(speaker)):
            if folder.is_dir():
                out += [p for p in sorted(folder.iterdir())
                        if p.suffix.lower() in AUDIO_SUFFIXES and p.suffix.lower() != ".wav"]
    return out


def convert_recordings_to_wav(corpus: "Corpus", log=print) -> dict:
    """Decode every non-WAV recording to 16-bit mono WAV at its own sample rate, next to the original
    name, and move the original to speakers/<spk>/_original_audio/ (kept, never deleted).
    Timing is unchanged, so existing alignments stay valid."""
    converted, skipped = 0, []
    for path in non_wav_recordings(corpus):
        target = path.with_suffix(".wav")
        if target.exists():
            skipped.append(f"{path.name}: {target.name} already exists; left as is")
            continue
        audio, rate = _load_mono(path)
        if not len(audio) or not np.isfinite(audio).all():
            skipped.append(f"{path.name}: could not decode")
            continue
        sf.write(target, audio, rate, subtype="PCM_16")
        keep = path.parents[1] / "_original_audio" / path.parent.name / path.name
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(keep))
        converted += 1
        log(f"converted {path.parent.name}/{path.name} -> {target.name}")
    return {"converted": converted, "skipped": skipped}


@dataclass
class AlignItem:
    source: str
    speaker: str
    sentence_id: str
    audio: Path
    text: str


def alignment_items(corpus: Corpus, sources: list[str], force: bool = False) -> list[AlignItem]:
    texts = corpus.text_of()
    items = []
    for source in sources:
        for speaker in corpus.speakers():
            if source == CLONE_REF:
                audio, text_file = corpus.audio(CLONE_REF, speaker, "passage"), corpus.clone_dir(speaker) / "passage.txt"
                if audio and text_file.is_file():
                    items.append(AlignItem(source, speaker, "passage", audio, text_file.read_text(encoding="utf-8")))
                continue
            for sid, text in texts.items():
                audio = corpus.audio(source, speaker, sid)
                if audio:
                    items.append(AlignItem(source, speaker, sid, audio, text))
    return [i for i in items if force or not corpus.textgrid(i.source, i.speaker, i.sentence_id).is_file()]


def run_mfa_batch(items: list[AlignItem], dest: callable, acoustic: str = DEFAULT_ACOUSTIC,
                  dictionary: str | None = DEFAULT_DICTIONARY, command: str = "mfa", jobs: int = 2,
                  extra_args: list[str] | None = None, word_lexicon: str | Path | None = None,
                  timeout_seconds: int = 3600, log=print, lexicon_copy: Path | None = None) -> dict:
    """Align many (audio, transcript) pairs in ONE `mfa align` call. Each (source, speaker) is its own MFA
    speaker, so speaker adaptation never mixes human and machine audio. Every source gets identical
    settings and the same lexicon. dest(item) gives the TextGrid destination."""
    executable = shutil.which(command)
    if not executable:
        raise RuntimeError("MFA command not found. Install with `conda install -c conda-forge montreal-forced-aligner` "
                           "(MFA 3.x, matching this acoustic model) or run inside the project's Docker image.")
    if not items:
        return {"requested": 0, "aligned": 0, "failed": [], "oov_words": []}
    phones = phone_inventory(acoustic, dictionary)
    if not phones:
        raise RuntimeError(f"Could not read the phone set from {acoustic}.")
    overrides = load_word_lexicon(word_lexicon, phones)
    with tempfile.TemporaryDirectory(prefix="naturalness_mfa_") as folder:
        root = Path(folder)
        corpus_dir, out_dir = root / "corpus", root / "aligned"
        groups, keys, vocabulary = {}, {}, set()
        for item in items:
            group = groups.setdefault((item.source, item.speaker), f"spk{len(groups):03d}")
            (corpus_dir / group).mkdir(parents=True, exist_ok=True)
            audio, sr = _load_mono(item.audio, MFA_SR)
            sf.write(corpus_dir / group / f"{item.sentence_id}.wav", audio, sr, subtype="PCM_16")
            words = mfa_text(item.text)
            vocabulary.update(words.split())
            (corpus_dir / group / f"{item.sentence_id}.lab").write_text(words, encoding="utf-8")
            keys[(group, item.sentence_id)] = item
        lexicon = root / "lexicon.dict"
        missing = write_lexicon(vocabulary, phones, lexicon, overrides)
        if lexicon_copy is not None:  # keep the exact pronunciations MFA used, for inspection
            Path(lexicon_copy).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(lexicon, lexicon_copy)
            Path(lexicon_copy).with_suffix(".phones.txt").write_text("\n".join(sorted(phones)) + "\n", encoding="utf-8")
        if missing:
            log(f"{len(missing)} word(s) have no akshara pronunciation and will align as OOV: {missing[:10]}")
        cmd = [executable, "align", str(corpus_dir), str(lexicon), str(acoustic), str(out_dir), "--clean",
               "-j", str(jobs), "--beam", "100", "--retry_beam", "400", *(extra_args or [])]
        log("Running: " + " ".join(cmd))
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"MFA exceeded the {timeout_seconds}-second timeout. Reduce corpus size or increase "
                "the timeout_seconds setting."
            ) from exc
        if proc.returncode:
            raise RuntimeError("MFA failed:\n" + (proc.stderr or proc.stdout)[-2000:])
        log(f"MFA finished in {time.perf_counter() - started:.0f} s")
        aligned, failed = 0, []
        found = {(p.parent.name, p.stem): p for p in out_dir.rglob("*.TextGrid")}
        for key, item in keys.items():
            if key in found:
                target = dest(item)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(found[key], target)
                aligned += 1
            else:
                failed.append(f"{item.source}/{item.speaker}/{item.sentence_id}")
    return {"requested": len(items), "aligned": aligned, "failed": failed, "oov_words": missing}


def align_corpus(corpus: Corpus, sources: list[str], force: bool = False, **mfa) -> dict:
    available = alignment_items(corpus, sources, force=True)
    if not available:
        raise ValueError("No matching audio files were found for the selected alignment sources.")
    items = available if force else [i for i in available if not corpus.textgrid(i.source, i.speaker, i.sentence_id).is_file()]
    mfa.setdefault("lexicon_copy", corpus.root / "alignments" / "_lexicon_used.dict")
    result = run_mfa_batch(items, lambda i: corpus.textgrid(i.source, i.speaker, i.sentence_id), **mfa)
    result["available"] = len(available)
    result["skipped_existing"] = len(available) - len(items)
    return result


def align_one(audio: np.ndarray, sr: int, transcript: str, **mfa) -> list[dict]:
    """Single-file alignment (used by Batch Research)."""
    with tempfile.TemporaryDirectory(prefix="naturalness_one_") as folder:
        wav, grid = Path(folder) / "sample.wav", Path(folder) / "sample.TextGrid"
        sf.write(wav, audio, sr, subtype="PCM_16")
        result = run_mfa_batch([AlignItem("x", "x", "sample", wav, transcript)], lambda _: grid, **mfa)
        if not grid.is_file():
            raise RuntimeError("MFA could not align this file. " + "; ".join(result["failed"]))
        return word_intervals(grid, len(audio) / sr)


def import_alignments(corpus: Corpus, source: str, folder: str | Path, force: bool = False) -> dict:
    """Copy TextGrids made by another aligner (e.g. a colleague's MFA setup) into the corpus.

    Expected names: <folder>/<speaker>/<sentence_id>.TextGrid (or <speaker>__<sentence_id>.TextGrid
    flat in <folder>). Each grid must have a words tier whose word sequence equals the stimulus text
    after the same normalization; otherwise it is rejected and listed, never silently used.
    Use ONE aligner for every source you compare: human and machine timings from different aligners
    are not comparable.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder.")
    if source != HUMAN and source != CLONE_REF:
        _validate_id(source.replace("+", "_"), "Source")
    texts, speakers = corpus.text_of(), set(corpus.speakers())
    imported, rejected, skipped = 0, [], 0
    for grid in sorted(folder.rglob("*.TextGrid")):
        if grid.parent != folder:
            speaker, sid = grid.parent.name, grid.stem
        elif "__" in grid.stem:
            speaker, sid = grid.stem.split("__", 1)
        else:
            rejected.append(f"{grid.name}: cannot tell the speaker (use <speaker>/<id>.TextGrid)")
            continue
        if speaker not in speakers:
            rejected.append(f"{grid.name}: unknown speaker {speaker!r}")
            continue
        if sid not in texts:
            rejected.append(f"{speaker}/{grid.name}: unknown sentence ID {sid!r}")
            continue
        try:
            words = word_intervals(grid)
        except Exception as exc:
            rejected.append(f"{speaker}/{sid}: unreadable TextGrid ({type(exc).__name__}: {exc})")
            continue
        got, want = [w["word"] for w in words], [w for w, _ in tokens_with_punct(texts[sid])]
        if [mfa_text(w) for w in got] != want:
            rejected.append(f"{speaker}/{sid}: word tier does not match the stimulus text")
            continue
        target = corpus.textgrid(source, speaker, sid)
        if target.is_file() and not force:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(grid, target)
        imported += 1
    return {"imported": imported, "skipped_existing": skipped, "rejected": rejected}


# ---------------------------------------------------------------- clone prompt


def make_clone_prompt(corpus: Corpus, speaker: str, min_s: float = 6.0, max_s: float = 12.0,
                      target_s: float = 9.0) -> tuple[Path, Path]:
    """Cut a 6-12 s prompt (audio + exact transcript) from the aligned cloning passage, starting and
    ending at pauses. Zero-shot cloners such as IndicF5 need a short prompt whose text matches exactly."""
    folder = corpus.clone_dir(speaker)
    wav_out, txt_out = folder / "prompt.wav", folder / "prompt.txt"
    grid = corpus.textgrid(CLONE_REF, speaker, "passage")
    if not grid.is_file():
        raise FileNotFoundError(f"Align the cloning passage first: `align --sources {CLONE_REF}`.")
    audio, sr = _load_mono(corpus.audio(CLONE_REF, speaker, "passage"))
    words = word_intervals(grid, len(audio) / sr)
    breaks = {0} | {i + 1 for i, w in enumerate(words) if w["pause_after_s"] >= PAUSE_MIN_S}
    best = None
    for a in sorted(breaks):
        for b in sorted(breaks):
            if b <= a or b > len(words):
                continue
            span = words[b - 1]["end_s"] - words[a]["start_s"]
            if min_s <= span <= max_s and not any(w["oov"] for w in words[a:b]):
                score = abs(span - target_s)
                best = (score, a, b) if best is None or score < best[0] else best
    if best is None:
        raise RuntimeError("No pause-bounded 6-12 s span in the passage; cut prompt.wav/prompt.txt by hand.")
    _, a, b = best
    start = max(words[a]["start_s"] - 0.10, 0.0)
    end = min(words[b - 1]["end_s"] + 0.15, len(audio) / sr)
    sf.write(wav_out, audio[int(start * sr): int(end * sr)], sr, subtype="PCM_16")
    txt_out.write_text(" ".join(w["word"] for w in words[a:b]), encoding="utf-8")
    return wav_out, txt_out


# ---------------------------------------------------------------- durations and the diff


def build_durations(corpus: Corpus, sources: list[str] | None = None) -> pd.DataFrame:
    """One row per aligned prosodic unit (word + its enclitics, see ENCLITICS) per source. Flags: ok,
    short, long, oov, mismatch (MFA word sequence differs from the stimulus), so bad alignments are
    excluded rather than averaged in."""
    texts, splits = corpus.text_of(), corpus.splits()
    sources = sources or [HUMAN, *corpus.systems()]
    rows = []
    for source in sources:
        for speaker in corpus.speakers():
            for sid, text in texts.items():
                grid = corpus.textgrid(source, speaker, sid)
                if not grid.is_file():
                    continue
                audio = corpus.audio(source, speaker, sid)
                seconds = _audio_info(audio)[1] if audio else None
                words, mismatch = sentence_units(grid, text, seconds)
                expected = unit_tokens(text)
                n = len(words)
                for w, (_, punct) in zip(words, expected if not mismatch else [("", "none")] * n):
                    # Compare in whole milliseconds: TextGrid times are decimals, and 1.26 - 1.23 can come out
                    # as 0.0299999, which flagged ~40% of words sitting exactly at MFA's 30 ms floor.
                    dur_ms = round(w["dur_s"] * 1000, 1)
                    long_ms = (MAX_WORD_S + LONG_PER_EXTRA_WORD_S * (w["n_tokens"] - 1)) * 1000
                    flag = ("mismatch" if mismatch else "oov" if w["oov"] else "short" if dur_ms < MIN_WORD_S * 1000
                            else "long" if dur_ms > long_ms else "ok")
                    rows.append({"source": source, "speaker": speaker, "sentence_id": sid,
                                 "split": corpus.split_of(speaker, sid, splits), **w,
                                 "n_words": n, "punct_after": punct, "flag": flag})
    columns = ["source", "speaker", "sentence_id", "split", "word_idx", "word", "start_s", "end_s",
               "dur_s", "pause_after_s", "n_phones", "oov", "n_tokens", "n_words", "punct_after", "flag"]
    frame = pd.DataFrame(rows, columns=columns)
    corpus.root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(corpus.durations_csv, index=False, encoding="utf-8")
    return frame


def pair_durations(durations: pd.DataFrame, system: str) -> pd.DataFrame:
    """Join human and machine words. diff_s = human - machine (the error fed back to the model).
    Sentences where any word is badly aligned on either side are dropped whole, so per-sentence
    rate normalization stays valid."""
    key = ["speaker", "sentence_id", "word_idx"]
    human = durations[durations.source == HUMAN]
    machine = durations[durations.source == system]
    if human.empty or machine.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    cols = ["word", "start_s", "end_s", "dur_s", "pause_after_s", "n_phones", "flag"]
    pairs = human[key + ["split", "n_words", "punct_after"] + cols].merge(
        machine[key + cols], on=key, suffixes=("_h", "_m"))
    pairs = pairs[pairs.word_h == pairs.word_m].rename(columns={"word_h": "word"}).drop(columns="word_m")
    bad = pairs[(pairs.flag_h != "ok") | (pairs.flag_m != "ok")][["speaker", "sentence_id"]].drop_duplicates()
    pairs = pairs.merge(bad, how="left", indicator=True)
    pairs = pairs[pairs._merge == "left_only"].drop(columns=["_merge", "flag_h", "flag_m"])
    pairs = pairs[pairs.groupby(["speaker", "sentence_id"]).word_idx.transform("size") == pairs.n_words].copy()
    pairs.insert(0, "system", system)
    pairs["diff_s"] = pairs.dur_s_h - pairs.dur_s_m
    pairs["log_ratio"] = np.log(pairs.dur_s_h / pairs.dur_s_m)
    pairs["log_ratio_resid"] = pairs.log_ratio - pairs.groupby(["speaker", "sentence_id"]).log_ratio.transform("mean")
    pairs["pause_diff_s"] = pairs.pause_after_s_h - pairs.pause_after_s_m
    return pairs.reset_index(drop=True)


def build_pairs(corpus: Corpus, durations: pd.DataFrame | None = None) -> pd.DataFrame:
    if durations is None:
        if not corpus.durations_csv.is_file():
            raise FileNotFoundError(f"{corpus.durations_csv} not found. Run `durations` after MFA alignment.")
        durations = pd.read_csv(corpus.durations_csv, keep_default_na=False)
    if durations.empty:
        pairs = pd.DataFrame(columns=PAIR_COLUMNS)
        corpus.pairs_csv.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(corpus.pairs_csv, index=False, encoding="utf-8")
        return pairs
    frames = [pair_durations(durations, s) for s in sorted(set(durations.source)) if s != HUMAN]
    pairs = (
        pd.concat([f for f in frames if len(f)], ignore_index=True)
        if any(len(f) for f in frames)
        else pd.DataFrame(columns=PAIR_COLUMNS)
    )
    corpus.pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(corpus.pairs_csv, index=False, encoding="utf-8")
    return pairs


def load_pairs(corpus: Corpus, system: str | None = None) -> pd.DataFrame:
    if not corpus.pairs_csv.is_file():
        raise FileNotFoundError(f"{corpus.pairs_csv} not found. Run `durations` after aligning human and TTS audio.")
    try:
        pairs = pd.read_csv(corpus.pairs_csv, keep_default_na=False, dtype={"speaker": str, "sentence_id": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    if pairs.empty:
        return pairs
    return pairs[pairs.system == system].reset_index(drop=True) if system else pairs


def alignment_coverage(corpus: Corpus, sources: list[str] | None = None,
                       durations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Report source audio, aligned files, and word-level failures so exclusions are measurable."""
    sources = sources or [HUMAN, *corpus.systems()]
    if durations is None and corpus.durations_csv.is_file():
        durations = pd.read_csv(corpus.durations_csv, keep_default_na=False)
    rows = []
    for source in sources:
        expected = sum(corpus.audio(source, speaker, sid) is not None
                       for speaker in corpus.speakers() for sid in corpus.text_of())
        aligned = sum(corpus.textgrid(source, speaker, sid).is_file()
                      for speaker in corpus.speakers() for sid in corpus.text_of())
        selected = durations[durations.source == source] if durations is not None and not durations.empty else pd.DataFrame()
        bad = selected[selected.flag != "ok"] if not selected.empty else pd.DataFrame()
        rows.append({
            "Source": source,
            "Audio files": expected,
            "Aligned files": aligned,
            "Alignment coverage (%)": 100 * aligned / expected if expected else float("nan"),
            "Aligned words": len(selected),
            "OOV words": int((selected.flag == "oov").sum()) if not selected.empty else 0,
            "Short words (<30 ms)": int((selected.flag == "short").sum()) if not selected.empty else 0,
            "Long words (>1.5 s)": int((selected.flag == "long").sum()) if not selected.empty else 0,
            "Mismatch words": int((selected.flag == "mismatch").sum()) if not selected.empty else 0,
            "Collapsed (<25 ms/akshara) %": round(100 * float(
                (selected.dur_s.astype(float) * 1000 / selected.n_phones.astype(float).clip(lower=1)
                 < COLLAPSE_MS_PER_PHONE).mean()), 1) if not selected.empty else float("nan"),
            "Sentences with flagged words": int(bad[["speaker", "sentence_id"]].drop_duplicates().shape[0])
            if not bad.empty else 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the duration-correction layer


FEATURES = ["m_log_dur", "m_pause", "m_sent_log_rate", "n_letters", "n_syll", "n_matra", "n_conjunct", "is_latin",
            "n_phones", "is_function", "rel_pos", "is_first", "is_last", "to_end", "sent_n_words", "is_question",
            "p_comma", "p_question", "p_final", "spk_offset"]


def features(frame: pd.DataFrame, speaker_offset: dict | None = None, default_offset: float = 0.0) -> pd.DataFrame:
    """Text + machine-alignment features. Uses only machine-side timings, so it applies to new sentences."""
    shapes = pd.DataFrame([word_shape(w) for w in frame.word], index=frame.index)
    group = frame.groupby(["speaker", "sentence_id"])
    n = frame.n_words.astype(float)
    last_punct = group.punct_after.transform("last")
    out = pd.DataFrame({
        "m_log_dur": np.log(frame.dur_s_m),
        "m_pause": frame.pause_after_s_m.where(frame.word_idx < n - 1, 0.0),
        "m_sent_log_rate": group.dur_s_m.transform(lambda d: np.log(d).mean()),
        **{c: shapes[c] for c in shapes},
        "n_phones": frame.n_phones_m.where(frame.n_phones_m > 0, shapes.n_letters),
        "is_function": frame.word.isin(FUNCTION_WORDS).astype(int),
        "rel_pos": np.where(n > 1, frame.word_idx / (n - 1).clip(lower=1), 0.0),
        "is_first": (frame.word_idx == 0).astype(int),
        "is_last": (frame.word_idx == n - 1).astype(int),
        "to_end": (n - 1 - frame.word_idx).clip(upper=5),
        "sent_n_words": n,
        "is_question": (last_punct == "question").astype(int),
        "p_comma": (frame.punct_after == "comma").astype(int),
        "p_question": (frame.punct_after == "question").astype(int),
        "p_final": (frame.punct_after == "final").astype(int),
        "spk_offset": frame.speaker.map(speaker_offset or {}).fillna(default_offset),
    }, index=frame.index)
    return out[FEATURES].astype(float)


class DurationModel:
    """Predicts log(human/machine) word duration and the human pause after each word.

    word kinds: none (leave machine durations unchanged; the baseline every layer must beat),
                global (one rate factor), bucket (mean per syllables x position x punctuation),
                ridge, gbm.  pause kinds: keep (machine pauses), learned (P(pause) x size).
    """

    WORD_KINDS = ("none", "global", "bucket", "ridge", "gbm")
    PAUSE_KINDS = ("keep", "learned")

    def __init__(self, word_kind: str = "ridge", pause_kind: str = "learned", seed: int = 0):
        self.word_kind, self.pause_kind, self.seed = word_kind, pause_kind, seed

    @staticmethod
    def _bucket(frame: pd.DataFrame, x: pd.DataFrame) -> pd.Series:
        syll = x.n_syll.clip(upper=4).astype(int).astype(str)
        pos = np.where(x.is_last == 1, "last", np.where(x.is_first == 1, "first", "mid"))
        return syll + "|" + pos + "|" + (frame.punct_after != "none").astype(int).astype(str)

    def fit(self, train: pd.DataFrame) -> "DurationModel":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.speaker_offset = train.groupby("speaker").log_ratio.mean().to_dict()
        self.default_offset = float(train.log_ratio.mean())
        x, y = features(train, self.speaker_offset), train.log_ratio.to_numpy()
        self.global_ = float(y.mean())
        if self.word_kind == "bucket":
            keys = self._bucket(train, x)
            stats = pd.DataFrame({"k": keys, "y": y}).groupby("k").y.agg(["mean", "size"])
            self.buckets_ = stats[stats["size"] >= 3]["mean"].to_dict()
        elif self.word_kind == "ridge":
            self.word_ = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))).fit(x, y)
        elif self.word_kind == "gbm":
            self.word_ = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                       min_samples_leaf=10, l2_regularization=1.0,
                                                       random_state=self.seed).fit(x, y)
        if self.pause_kind == "learned":
            inner = x.is_last == 0
            xp, pause = x[inner], train.pause_after_s_h[inner].to_numpy()
            has = pause >= PAUSE_MIN_S
            if has.all() or not has.any():
                self.pause_p_ = float(has.mean())
                self.pause_clf_ = None
            else:
                self.pause_clf_ = (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, min_samples_leaf=10,
                                                                  random_state=self.seed) if len(xp) >= 200
                                   else make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))).fit(xp, has)
            self.pause_size_ = (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))).fit(
                xp[has], np.log(pause[has])) if has.sum() >= 5 else None)
            self.pause_median_ = float(np.median(pause[has])) if has.any() else 0.0
            # Typical human gap between words that are NOT a pause (closures, micro-gaps).
            self.short_gap_ = float(np.median(pause[~has])) if (~has).any() else 0.0
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Columns pred_log_ratio, pred_dur_s, pred_pause_s for machine-aligned words."""
        x = features(frame, self.speaker_offset, self.default_offset)
        if self.word_kind == "none":
            log_ratio = np.zeros(len(x))
        elif self.word_kind == "global":
            log_ratio = np.full(len(x), self.global_)
        elif self.word_kind == "bucket":
            log_ratio = self._bucket(frame, x).map(self.buckets_).fillna(self.global_).to_numpy()
        else:
            log_ratio = self.word_.predict(x)
        pause = frame.pause_after_s_m.to_numpy(float).copy()
        if self.pause_kind == "learned":
            inner = (x.is_last == 0).to_numpy()
            if inner.any():
                xi = x[inner]
                p = self.pause_clf_.predict_proba(xi)[:, 1] if self.pause_clf_ is not None else np.full(len(xi), self.pause_p_)
                size = np.exp(self.pause_size_.predict(xi)) if self.pause_size_ is not None else np.full(len(xi), self.pause_median_)
                # No pause predicted: keep the machine's micro-gap if it is already below the pause
                # threshold, otherwise shrink it to the typical human non-pause gap. Setting it to 0
                # would also delete natural closures between words.
                machine_gap = pause[inner]
                short = np.where(machine_gap < PAUSE_MIN_S, machine_gap,
                                 getattr(self, "short_gap_", 0.0))
                pause[inner] = np.where(p >= 0.5, np.clip(size, PAUSE_MIN_S, MAX_PAUSE_S), short)
        return pd.DataFrame({"pred_log_ratio": log_ratio, "pred_dur_s": frame.dur_s_m.to_numpy() * np.exp(log_ratio),
                             "pred_pause_s": pause}, index=frame.index)

    def save(self, path: Path, meta: dict | None = None):
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        self.meta = meta or {}
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "DurationModel":
        import joblib
        return joblib.load(path)


def _rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def train_duration_model(pairs: pd.DataFrame, seed: int = 0) -> tuple[DurationModel, pd.DataFrame]:
    """Fit every word x pause model on train, choose the best on dev. Test is never touched here."""
    if pairs.empty:
        raise ValueError("There are no valid word pairs to train on. Align audio and rebuild the duration tables.")
    required = {
        "split", "speaker", "sentence_id", "word_idx", "n_words", "word", "dur_s_h", "dur_s_m",
        "pause_after_s_h", "pause_after_s_m", "log_ratio", "punct_after",
    }
    if not required.issubset(pairs.columns):
        raise ValueError(f"Duration pairs are missing required columns: {', '.join(sorted(required - set(pairs.columns)))}.")
    train, dev = pairs[pairs.split == "train"], pairs[pairs.split == "dev"]
    if len(train) < 20 or train[["speaker", "sentence_id"]].drop_duplicates().shape[0] < 5:
        raise ValueError(f"Too little aligned training data ({len(train)} words). Record and align more train sentences.")
    if dev.empty:
        raise ValueError("No aligned dev sentences; model selection needs the dev split.")
    rows, models = [], {}
    inner = dev.word_idx < dev.n_words - 1
    for wk in DurationModel.WORD_KINDS:
        for pk in DurationModel.PAUSE_KINDS:
            model = DurationModel(wk, pk, seed).fit(train)
            pred = model.predict(dev)
            rows.append({"word_model": wk, "pause_model": pk,
                         "dev_word_rmse_ms": 1000 * _rmse(pred.pred_dur_s, dev.dur_s_h),
                         "dev_log_rmse": _rmse(pred.pred_log_ratio, dev.log_ratio),
                         "dev_pause_rmse_ms": 1000 * _rmse(pred.pred_pause_s[inner], dev.pause_after_s_h[inner]),
                         "machine_word_rmse_ms": 1000 * _rmse(dev.dur_s_m, dev.dur_s_h),
                         "machine_pause_rmse_ms": 1000 * _rmse(dev.pause_after_s_m[inner], dev.pause_after_s_h[inner])})
            models[(wk, pk)] = model
    report = pd.DataFrame(rows)
    # Ties go to the simpler model: WORD_KINDS is ordered from simplest ("none") upwards.
    order = {kind: i for i, kind in enumerate(DurationModel.WORD_KINDS)}
    by_word = report.groupby("word_model").dev_log_rmse.min().round(6)
    best_word = min(by_word.index, key=lambda k: (by_word[k], order[k]))
    candidates = report[report.word_model == best_word]
    pause_order = {kind: i for i, kind in enumerate(DurationModel.PAUSE_KINDS)}
    best_pause = min(candidates.itertuples(),
                     key=lambda r: (round(float(np.nan_to_num(r.dev_pause_rmse_ms, nan=np.inf)), 3),
                                    pause_order[r.pause_model])).pause_model
    report["selected"] = (report.word_model == best_word) & (report.pause_model == best_pause)
    # Refit the chosen configuration on train + dev; the test split is still never seen.
    fit_on = pd.concat([train, dev])
    final = DurationModel(best_word, best_pause, seed).fit(fit_on)
    final.trained_on = {"words": len(fit_on), "sentences": int(fit_on.sentence_id.nunique()),
                        "splits": ["train", "dev"], "speakers": sorted(fit_on.speaker.unique().tolist()),
                        "beats_no_correction_on_dev": bool(best_word != "none" or best_pause != "keep")}
    return final, report


# ---------------------------------------------------------------- applying the correction


def _stretch(segment: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Pitch-preserving time-stretch with Praat overlap-add."""
    if abs(factor - 1.0) < 0.02 or len(segment) < int(0.04 * sr):
        return segment
    import parselmouth
    from parselmouth.praat import call
    sound = parselmouth.Sound(segment.astype(np.float64), sampling_frequency=sr)
    return call(sound, "Lengthen (overlap-add)", 75, 600, float(factor)).values[0].astype(np.float32)


def _resize_pause(segment: np.ndarray, target: int) -> np.ndarray:
    """Shorten by removing the middle, lengthen by inserting silence in the middle; edges keep the
    original release and onset."""
    if target <= 0:
        return segment[:0]
    if len(segment) >= target:
        head = target // 2
        return np.concatenate([segment[:head], segment[len(segment) - (target - head):]])
    head = len(segment) // 2
    return np.concatenate([segment[:head], np.zeros(target - len(segment), np.float32), segment[head:]])


def _append(out: list[np.ndarray], piece: np.ndarray, fade: int):
    if not out or not len(piece) or len(out[-1]) < fade or len(piece) < fade:
        out.append(piece)
        return
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    prev = out[-1]
    prev[-fade:] = prev[-fade:] * (1 - ramp) + piece[:fade] * ramp
    out.append(piece[fade:])


def apply_durations(audio: np.ndarray, sr: int, words: list[dict], target_dur: list[float],
                    target_pause: list[float], limits: tuple[float, float] = STRETCH_LIMITS) -> tuple[np.ndarray, dict]:
    """Re-time machine speech: each word to its target duration (factor clamped to `limits`), each
    inter-word gap to its target pause. Leading and trailing silence are kept."""
    fade = int(FADE_S * sr)
    out, factors = [], []
    cursor = int(round(words[0]["start_s"] * sr)) if words else len(audio)
    out.append(audio[:cursor].copy())
    for i, w in enumerate(words):
        a, b = int(round(w["start_s"] * sr)), int(round(w["end_s"] * sr))
        factor = float(np.clip((target_dur[i] + FADE_S) / max(w["dur_s"], 1e-3), *limits))
        factors.append(factor)
        _append(out, _stretch(audio[a:b].copy(), sr, factor), fade)
        if i + 1 < len(words):
            nxt = int(round(words[i + 1]["start_s"] * sr))
            target = int(round(float(np.clip(target_pause[i], 0.0, MAX_PAUSE_S)) * sr))
            _append(out, _resize_pause(audio[b:nxt].copy(), target), fade)
        cursor = b
    _append(out, audio[cursor:].copy(), fade)
    y = np.concatenate(out).astype(np.float32)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.99:
        y = y / peak * 0.99
    return y, {"mean_factor": float(np.mean(factors)) if factors else 1.0,
               "clamped": int(sum(f in limits for f in factors)), "words": len(words)}


def correct_system(corpus: Corpus, system: str, splits: tuple[str, ...] = ("dev", "test"),
                   oracle: bool = False, log=print) -> str:
    """Write generated/<system>+dur (or +oracle) for the chosen splits. +oracle uses the real human
    durations: a ceiling for analysis only, never a result."""
    out_system = f"{system}+{'oracle' if oracle else 'dur'}"
    model = None if oracle else DurationModel.load(corpus.model_path(system))
    pairs = load_pairs(corpus, system)
    durations = pd.read_csv(corpus.durations_csv, keep_default_na=False, dtype={"speaker": str, "sentence_id": str})
    texts, split_map = corpus.text_of(), corpus.splits()
    written = 0
    for speaker in corpus.speakers():
        for sid in texts:
            if corpus.split_of(speaker, sid, split_map) not in splits:
                continue
            audio_path = corpus.audio(system, speaker, sid)
            grid = corpus.textgrid(system, speaker, sid)
            if not audio_path or not grid.is_file():
                continue
            audio, sr = _load_mono(audio_path)
            words, mismatch = sentence_units(grid, texts[sid], len(audio) / sr)
            if mismatch:
                log(f"skip {speaker}/{sid}: machine alignment does not match the text")
                continue
            if oracle:
                rows = pairs[(pairs.speaker == speaker) & (pairs.sentence_id == sid)].sort_values("word_idx")
                if len(rows) != len(words):
                    continue
                target_dur, target_pause = rows.dur_s_h.tolist(), rows.pause_after_s_h.tolist()
            else:
                m = durations[(durations.source == system) & (durations.speaker == speaker) & (durations.sentence_id == sid)]
                if m.empty or (m.flag == "mismatch").any():
                    log(f"skip {speaker}/{sid}: machine alignment does not match the text")
                    continue
                frame = m.sort_values("word_idx").rename(columns={"dur_s": "dur_s_m", "pause_after_s": "pause_after_s_m",
                                                                  "n_phones": "n_phones_m"})
                pred = model.predict(frame.reset_index(drop=True))
                target_dur, target_pause = pred.pred_dur_s.tolist(), pred.pred_pause_s.tolist()
            y, info = apply_durations(audio, sr, words, target_dur, target_pause)
            path = corpus.generated_path(out_system, speaker, sid)
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, y, sr, subtype="PCM_16")
            written += 1
    log(f"{out_system}: wrote {written} files")
    return out_system


# ---------------------------------------------------------------- generation


STOCK_SYSTEMS = ("bulbul-v3", "mms-tts-hin", "mms-tts-hin-ft", "maya2-native")
CLONE_SYSTEMS = ("clone-indicf5", "clone-xtts")


def _stock_tts(
    system: str,
    text: str,
    device: str = "cpu",
    mms_finetune_checkpoint: Path | None = None,
) -> tuple[np.ndarray, int]:
    if system == "bulbul-v3":
        import io

        from speech_core import synthesize
        wav, _, _ = synthesize(text)
        audio, sr = sf.read(io.BytesIO(wav), dtype="float32")
        return audio, sr
    if system == "mms-tts-hin-ft":
        if mms_finetune_checkpoint is None:
            raise ValueError("Provide the corpus MMS duration fine-tuning checkpoint.")
        import mms_finetune

        return mms_finetune.synthesize(text, device, mms_finetune_checkpoint)
    import research
    if system == "mms-tts-hin":
        audio, sr, _ = research.tts_mms(text, device=device)
    elif system == "maya2-native":
        import torch
        torch.manual_seed(research.TTS_SEED)  # Maya samples; a fixed seed keeps regenerations repeatable
        audio, sr, _ = research.tts_maya(text, voice=os.getenv("MAYA_VOICE", "Ahana"))
    else:
        raise ValueError(f"Unknown system {system}")
    return audio, sr


def generate(corpus: Corpus, system: str, splits: tuple[str, ...] = SPLITS, force: bool = False,
             device: str = "cpu", speaker_consent: bool = False, log=print) -> int:
    """Synthesize every stimulus for every speaker. Stock voices are generated once per sentence and
    copied to each speaker folder; cloned voices are conditioned only on that speaker's clone prompt."""
    texts, split_map = corpus.text_of(), corpus.splits()
    speakers = corpus.speakers()
    if system in CLONE_SYSTEMS and not speaker_consent:
        raise ValueError(
            "Voice cloning requires confirmation that every recorded speaker gave explicit permission. "
            "Pass speaker_consent=True only after obtaining that permission."
        )
    todo = {spk: [sid for sid in texts if corpus.split_of(spk, sid, split_map) in splits
                  and (force or not corpus.generated_path(system, spk, sid).is_file())] for spk in speakers}
    written = 0
    if system in CLONE_SYSTEMS:
        import voice_clone
        issues = validate_corpus(corpus)
        if has_errors(issues):
            raise RuntimeError("Refusing to clone: " + "; ".join(i["message"] for i in issues if i["level"] == "error"))
        for spk, sids in todo.items():
            if not sids:
                continue
            folder = corpus.clone_dir(spk)
            prompt_wav, prompt_txt = folder / "prompt.wav", folder / "prompt.txt"
            if not prompt_wav.is_file() or not prompt_txt.is_file():
                prompt_wav, prompt_txt = make_clone_prompt(corpus, spk)
            jobs = [{"text": texts[sid], "out": str(corpus.generated_path(system, spk, sid))} for sid in sids]
            written += voice_clone.generate_batch(system.removeprefix("clone-"), str(prompt_wav),
                                                  prompt_txt.read_text(encoding="utf-8").strip(), jobs,
                                                  passage_wav=str(corpus.audio(CLONE_REF, spk, "passage") or prompt_wav),
                                                  # "cpu" is the CLI/UI default, so treat it as "auto" (GPU when present)
                                                  device=None if device == "cpu" else device,
                                                  log=log)
        return written
    mms_finetune_checkpoint = None
    if system == "mms-tts-hin-ft":
        mms_finetune_checkpoint = corpus.mms_finetune_path
        if not mms_finetune_checkpoint.is_file():
            raise FileNotFoundError(
                f"{mms_finetune_checkpoint} not found. Run `mms-finetune` before generating this system."
            )
    cache: dict[str, Path] = {}
    for spk, sids in todo.items():
        for sid in sids:
            target = corpus.generated_path(system, spk, sid)
            target.parent.mkdir(parents=True, exist_ok=True)
            if sid in cache:
                shutil.copyfile(cache[sid], target)
            else:
                audio, sr = _stock_tts(
                    system,
                    texts[sid],
                    device,
                    mms_finetune_checkpoint if system == "mms-tts-hin-ft" else None,
                )
                sf.write(target, np.asarray(audio, np.float32), sr, subtype="PCM_16")
                cache[sid] = target
                log(f"{system}: {sid}")
            written += 1
    return written


# ---------------------------------------------------------------- evaluation


def duration_metrics(pairs: pd.DataFrame) -> dict:
    """Machine-vs-human timing on matched words. rhythm_corr removes each sentence's overall rate, so it
    measures the word-to-word timing pattern alone."""
    if pairs.empty:
        return {}
    inner = pairs.word_idx < pairs.n_words - 1
    g = pairs.groupby(["speaker", "sentence_id"])
    resid_h = np.log(pairs.dur_s_h) - g.dur_s_h.transform(lambda d: np.log(d).mean())
    resid_m = np.log(pairs.dur_s_m) - g.dur_s_m.transform(lambda d: np.log(d).mean())
    span = lambda s: (g[f"end_s_{s}"].max() - g[f"start_s_{s}"].min())  # noqa: E731
    ph, pm = pairs.pause_after_s_h[inner] >= PAUSE_MIN_S, pairs.pause_after_s_m[inner] >= PAUSE_MIN_S
    tp = float((ph & pm).sum())
    f1 = 2 * tp / (ph.sum() + pm.sum()) if (ph.sum() + pm.sum()) else float("nan")
    return {
        "sentences": int(g.ngroups), "words": int(len(pairs)),
        "word_rmse_ms": 1000 * _rmse(pairs.dur_s_m, pairs.dur_s_h),
        "word_mae_ms": 1000 * float(np.mean(np.abs(pairs.diff_s))),
        "mean_diff_ms (human-machine)": 1000 * float(pairs.diff_s.mean()),
        "log_dur_corr": _corr(np.log(pairs.dur_s_m), np.log(pairs.dur_s_h)),
        "rhythm_corr": _corr(resid_m, resid_h),
        "pause_rmse_ms": 1000 * _rmse(pairs.pause_after_s_m[inner], pairs.pause_after_s_h[inner]),
        "pause_f1": f1,
        "speech_span_ratio (machine/human)": float(np.median(span("m") / span("h"))),
    }


def bootstrap_metric(pairs: pd.DataFrame, metric: str, reps: int = 500, seed: int = 7) -> tuple[float, float]:
    """95% CI resampling whole sentences (words within a sentence are not independent)."""
    groups = [g for _, g in pairs.groupby(["speaker", "sentence_id"])]
    if len(groups) < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        sample = pd.concat([groups[i].assign(sentence_id=f"{groups[i].sentence_id.iloc[0]}#{k}")
                            for k, i in enumerate(rng.integers(0, len(groups), len(groups)))])
        draws.append(duration_metrics(sample)[metric])
    return float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))


def evaluate(corpus: Corpus, systems: list[str], split: str = "test", acoustic: bool = False,
             ci: bool = True, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report one row per system on one split, on the sentences EVERY listed system has valid pairs for
    (a system that fails to align some sentences must not be scored on an easier subset). Oracle
    systems are labelled so they are never mistaken for results."""
    if split not in SPLITS:
        raise ValueError(f"Unknown evaluation split {split!r}; choose from {', '.join(SPLITS)}.")
    if not systems:
        raise ValueError("Select at least one system to evaluate.")
    if len(set(systems)) != len(systems):
        raise ValueError("Each system may be selected only once.")
    pairs = load_pairs(corpus)
    if pairs.empty:
        raise ValueError("There are no valid human/TTS word pairs. Align both sources and rebuild durations first.")
    required = {"system", "split", "speaker", "sentence_id"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"Duration pairs are missing required columns: {', '.join(sorted(required - set(pairs.columns)))}.")
    pairs = pairs[(pairs.split == split) & pairs.system.isin(systems)]
    present = [s for s in systems if (pairs.system == s).any()]
    missing = [system for system in systems if system not in present]
    if missing:
        raise ValueError(
            f"No valid {split} pairs for: {', '.join(missing)}. "
            "Every selected system must have valid alignments on this split."
        )
    keys = pairs.groupby(["speaker", "sentence_id"]).system.nunique()
    common = keys[keys == len(present)].index
    dropped = len(keys) - len(common)
    if dropped:
        log(f"{dropped} {split} sentence(s) lack a valid alignment for some system and are excluded for all")
    if not len(common):
        raise ValueError(
            f"No common {split} sentences have valid alignments for all selected systems. "
            "Check word coverage and alignment flags before evaluating."
        )
    pairs = pairs.set_index(["speaker", "sentence_id"]).loc[common].reset_index() if len(common) else pairs.iloc[:0]
    rows, words = [], []
    for system in present:
        p = pairs[pairs.system == system]
        if p.empty:
            continue
        row = {"system": system, "split": split, "oracle": system.endswith("+oracle"), **duration_metrics(p)}
        if ci:
            for metric in ("word_rmse_ms", "rhythm_corr"):
                row[f"{metric} 95% CI"] = "[{:.3g}, {:.3g}]".format(*bootstrap_metric(p, metric))
        if acoustic:
            row |= acoustic_metrics(corpus, system, p, log=log)
        rows.append(row)
        words.append(p)
    report = pd.DataFrame(rows)
    out = corpus.root / "eval"
    out.mkdir(exist_ok=True)
    if len(report):
        report.to_csv(out / f"report_{split}.csv", index=False)
        pd.concat(words).to_csv(out / f"words_{split}.csv", index=False, encoding="utf-8")
    return report, pd.concat(words) if words else pd.DataFrame()


def _word_f0(audio: np.ndarray, sr: int, words: list[dict]) -> np.ndarray:
    """Median F0 per word in semitones relative to the utterance median (NaN when unvoiced)."""
    import parselmouth
    pitch = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr).to_pitch(time_step=0.01,
                                                                                          pitch_floor=75, pitch_ceiling=600)
    f0, t = pitch.selected_array["frequency"], pitch.xs()
    voiced = f0 > 0
    if voiced.sum() < 5:
        return np.full(len(words), np.nan)
    ref = np.median(f0[voiced])
    out = []
    for w in words:
        sel = voiced & (t >= w["start_s"]) & (t <= w["end_s"])
        out.append(12 * np.log2(np.median(f0[sel]) / ref) if sel.sum() >= 2 else np.nan)
    return np.array(out)


_SV = {}


def speaker_similarity(a: np.ndarray, b: np.ndarray, sr: int = MFA_SR) -> float:
    """Cosine similarity of WavLM speaker-verification x-vectors (a real speaker-identity embedding)."""
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector
    if not _SV:
        _SV["fe"] = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        _SV["model"] = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
    inputs = _SV["fe"]([a, b], sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.inference_mode():
        emb = torch.nn.functional.normalize(_SV["model"](**inputs).embeddings, dim=-1)
    return float((emb[0] * emb[1]).sum())


def acoustic_metrics(corpus: Corpus, system: str, pairs: pd.DataFrame, log=print) -> dict:
    """Word-level F0 correlation, speaker similarity to the real speaker, UTMOS and synthetic-detector
    score. Each metric is optional: a missing model gives NaN and a log line, not a crash."""
    f0_h, f0_m, sims, mos, det = [], [], [], [], []
    errors = set()
    for (speaker, sid), group in pairs.groupby(["speaker", "sentence_id"]):
        human, sr_h = _load_mono(corpus.audio(HUMAN, speaker, sid), MFA_SR)
        machine, sr_m = _load_mono(corpus.audio(system, speaker, sid), MFA_SR)
        hw = word_intervals(corpus.textgrid(HUMAN, speaker, sid), len(human) / sr_h)
        mw = word_intervals(corpus.textgrid(system, speaker, sid), len(machine) / sr_m)
        if len(hw) == len(mw):
            f0_h.extend(_word_f0(human, sr_h, hw)); f0_m.extend(_word_f0(machine, sr_m, mw))
        for name, fn, sink in (("speaker_sim", lambda: speaker_similarity(human, machine), sims),
                               ("utmos", lambda: _utmos(machine), mos),
                               ("detector", lambda: _detector(machine), det)):
            if name in errors:
                continue
            try:
                sink.append(fn())
            except Exception as exc:  # optional metric
                errors.add(name)
                log(f"{name} unavailable: {type(exc).__name__}: {exc}")
    f0_h, f0_m = np.array(f0_h), np.array(f0_m)
    ok = np.isfinite(f0_h) & np.isfinite(f0_m)
    mean = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    return {"word_f0_corr": _corr(f0_h[ok], f0_m[ok]), "speaker_sim": mean(sims), "utmos": mean(mos),
            "synthetic_prob": mean(det)}


def _utmos(y16: np.ndarray) -> float:
    from research import predicted_utmos
    return float(predicted_utmos(y16, MFA_SR))


def _detector(y16: np.ndarray) -> float:
    from deepfake import prepare, zero_shot_summary
    return float(zero_shot_summary(prepare(y16, MFA_SR))["zero_shot_mean"])


# ---------------------------------------------------------------- listening test


LISTEN_SR = 16_000
LISTEN_RMS_DBFS = -23.0


def _listening_clip(path: Path, grid: Path) -> np.ndarray:
    """Same treatment for human and machine: 16 kHz (no bandwidth cue), 150 ms margins around the
    first/last word (no silence cue), equal loudness (no level cue)."""
    y, sr = _load_mono(path, LISTEN_SR)
    words = word_intervals(grid, len(y) / sr)
    if words:
        y = y[max(int((words[0]["start_s"] - 0.15) * sr), 0): int((words[-1]["end_s"] + 0.15) * sr)]
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64)) + 1e-12))
    y = y * (10 ** (LISTEN_RMS_DBFS / 20) / max(rms, 1e-8))
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    return (y / peak * 0.95 if peak > 0.95 else y).astype(np.float32)


def build_listening_test(corpus: Corpus, systems: list[str], split: str = "test", seed: int = 11) -> pd.DataFrame:
    """Two-alternative forced choice: each trial pairs the human and one system on the same test
    sentence, with the human's A/B position randomized and balanced."""
    out = corpus.root / "listening"
    if (out / "responses.csv").is_file():
        raise FileExistsError("listening/responses.csv exists; building a new kit would orphan those responses.")
    systems = [s for s in systems if not s.endswith("+oracle")]
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    split_map, rows = corpus.splits(), []
    for speaker in corpus.speakers():
        for sid in corpus.text_of():
            if corpus.split_of(speaker, sid, split_map) != split:
                continue
            human = corpus.audio(HUMAN, speaker, sid)
            if not human or not corpus.textgrid(HUMAN, speaker, sid).is_file():
                continue
            for system in systems:
                machine = corpus.audio(system, speaker, sid)
                if not machine or not corpus.textgrid(system, speaker, sid).is_file():
                    continue
                trial = f"t{len(rows):04d}"
                human_first = bool(rng.integers(0, 2))
                # Clip names carry only the A/B position, never "human"/"machine": the kit may be
                # copied to listeners' machines, where a filename would give the answer away.
                order = ((HUMAN, human), (system, machine)) if human_first else ((system, machine), (HUMAN, human))
                names = []
                for position, (src, path) in zip("AB", order):
                    clip = out / "clips" / f"{trial}_{position}.wav"
                    sf.write(clip, _listening_clip(path, corpus.textgrid(src, speaker, sid)), LISTEN_SR, subtype="PCM_16")
                    names.append(clip.name)
                rows.append({"trial_id": trial, "speaker": speaker, "sentence_id": sid, "system": system,
                             "clip_a": names[0], "clip_b": names[1], "human_is": "A" if human_first else "B"})
    trials = pd.DataFrame(rows)
    trials.to_csv(out / "trials.csv", index=False)
    return trials


def listener_trials(trials: pd.DataFrame, listener: str) -> pd.DataFrame:
    """Latin square: each listener hears each sentence once, under one system, so familiarity with a
    sentence's human version cannot help on later trials. Order is shuffled per listener."""
    systems = sorted(trials.system.unique())
    offset = int(hashlib.sha256(listener.strip().casefold().encode()).hexdigest(), 16) % max(len(systems), 1)
    sentences = sorted(trials[["speaker", "sentence_id"]].drop_duplicates().itertuples(index=False))
    chosen = []
    for k, (speaker, sid) in enumerate(sentences):
        options = trials[(trials.speaker == speaker) & (trials.sentence_id == sid)]
        wanted = systems[(k + offset) % len(systems)]
        pick = options[options.system == wanted]
        chosen.append(pick if len(pick) else options.iloc[[k % len(options)]])
    picked = pd.concat(chosen) if chosen else trials.iloc[:0]
    seed = int(hashlib.sha256(listener.encode()).hexdigest()[:8], 16)
    return picked.sample(frac=1, random_state=seed).reset_index(drop=True)


RESPONSE_COLUMNS = ["time", "listener", "trial_id", "system", "speaker", "sentence_id", "answer", "human_is",
                    "correct", "response_s"]


def record_response(corpus: Corpus, row: dict):
    """Append one answer. A fixed column order keeps the append-only CSV parseable whatever the caller sends."""
    unknown = set(row) - set(RESPONSE_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown response fields: {sorted(unknown)}")
    path = corpus.root / "listening" / "responses.csv"
    pd.DataFrame([row]).reindex(columns=RESPONSE_COLUMNS).to_csv(path, mode="a", header=not path.is_file(), index=False)


def listening_report(corpus: Corpus) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per system: how often listeners picked the real human. 50% = indistinguishable from human.
    Wilson 95% CI and an exact two-sided binomial test against 50%."""
    from scipy.stats import binomtest
    path = corpus.root / "listening" / "responses.csv"
    if not path.is_file():
        return pd.DataFrame(), pd.DataFrame()
    responses = pd.read_csv(path, dtype={"listener": str})
    responses = responses.drop_duplicates(["listener", "trial_id"], keep="first")
    rows = []
    for system, g in responses.groupby("system"):
        n, k = len(g), int(g.correct.sum())
        p = k / n
        z = 1.96
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        rows.append({"system": system, "trials": n, "listeners": g.listener.nunique(),
                     "human_identified": p, "ci_low": centre - half, "ci_high": centre + half,
                     "p_vs_chance": binomtest(k, n, 0.5).pvalue,
                     "verdict": ("distinguishable (listeners find the human)" if centre - half > 0.5
                                 else "machine judged MORE human" if centre + half < 0.5
                                 else "not distinguishable at this sample size")})
    per_listener = responses.groupby(["listener", "system"]).correct.agg(["size", "mean"]).reset_index()
    return pd.DataFrame(rows), per_listener
