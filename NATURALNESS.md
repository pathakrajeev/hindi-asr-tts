# Naturalness pipeline: making TTS timing human-like

This pipeline implements the five action items:

| Action item | Where |
|---|---|
| MFA word durations for human and machine speech | `naturalness.align_corpus`, `build_durations` |
| Duration diff (human − machine) fed back into a correction layer | `build_pairs`, `DurationModel`, `correct_system` |
| Formal train/test evaluation | `make_splits`, `evaluate`, listening test |
| Voice cloning at the generation stage | `voice_clone.py`, `generate --system clone-*` |
| Separate read-aloud sample for cloning; stimuli only for testing | `validate_corpus` leakage check (blocks cloning) |

Code: `naturalness.py` (all logic), `naturalness_cli.py` (commands below), `naturalness_ui.py`
(the **Naturalness** mode in the app), `voice_clone.py` (cloning backends), `tests/`.

## Hugging Face Spaces (Docker)

The Docker image sets `NATURALNESS_ROOT=/data/naturalness` and includes the MMS fine-tuning
module. Hugging Face Space disk is ephemeral by default, so attach a **read-write Storage Bucket**
volume at `/data` in the Space settings before collecting recordings or training. If you mount the
bucket elsewhere, set the Space variable `NATURALNESS_ROOT` to a writable folder inside that mount.
Otherwise corpus recordings, split files, and fine-tuned checkpoints can be lost when the Space
restarts.

Select a CUDA-enabled Space for practical training speed; the fine-tuner also accepts `--device
cpu`, but CPU training can be slow. The app runs on port `7860`. MMS weights download from the Hub
on first use, so allow outbound access and enough disk for the cache and dataset. Do not put
speaker recordings or model checkpoints into the Git repository.

## 1. Prepare the data

1. Write `sentences.txt` with one stimulus sentence per line (100–200 sentences). Mix statements,
   questions and lists. Spell numbers out as words.
2. Create the corpus and fix the split. **Do this before any analysis, and never edit the split
   afterwards.**
   ```bash
   python naturalness_cli.py init --stimuli sentences.txt --speakers S01
   python naturalness_cli.py split            # 70/15/15 by sentence
   # with several speakers, optionally hold one out entirely:
   # python naturalness_cli.py split --test-speakers S03
   ```
   Or use **Naturalness → Pipeline → Create corpus and fixed splits** in the app.
   `init --force` only replaces a corpus before splits, alignments, generated audio, or duration
   tables exist. Once analysis has started, create a new corpus folder instead of overwriting it.
   At least three unique stimuli are required for non-empty train/dev/test splits. If corpus setup
   was interrupted before splits were written, the app lets you append more unique stimuli while
   preserving existing IDs; after recordings or analysis begin, restore the corpus or start fresh
   rather than changing the stimulus list.
3. Record each speaker:
   - `data/naturalness/speakers/S01/clone_ref/passage.wav` plus `passage.txt`: **2–5 minutes**
     of read-aloud text (a story, a news paragraph). It must share **no sentence** with the stimuli.
   - `data/naturalness/speakers/S01/stimuli/sent_001.wav`, … one file per stimulus.
   - Use the same microphone and room throughout, 16 kHz or higher, mono, no clipping.
   The app can import files named `S01__sent_001.wav`, `S01__passage.wav`, and
   `S01__passage.txt`. Optionally include `S01__sent_001.txt` to verify that the recording transcript
   matches the frozen stimulus; this check file is not stored. Passage audio and transcript must be
   uploaded together. Existing files are never overwritten.
4. Check the corpus. The check exits with an error if the passage leaks stimulus text, meaning a
   run of 4+ consecutive words is shared with a stimulus, or if a passage audio file is a copy of a
   stimulus recording.
   ```bash
   python naturalness_cli.py check
   ```

## 2. MFA alignment

MFA 3.x is required (the IndicMFA Hindi model is version 3.0.7). The Docker image installs it.
Locally, use `conda install -c conda-forge montreal-forced-aligner`.

**Lexicon note:** `Hindi_Dict_g2g.txt` lists akshara units only (`का`, `क्ष`, `्`), not words.
Passing it to MFA directly makes every real word out-of-vocabulary. The pipeline reads the
akshara phone set from the acoustic model and writes a word-level lexicon for every run
(`मौसम → मौ स म`). Words with no Devanagari, such as Latin-script English, still align as OOV
unless a supplemental pronunciation is supplied. OOVs, mismatches, and suspiciously short/long
intervals are reported; any sentence with a flagged word is excluded from paired duration
modeling. Do not treat dropped sentences as valid coverage.

Create a UTF-8 supplemental lexicon with one normalized word and acoustic-model phone sequence
per line (the example phone sequences must be verified against the selected acoustic model):

```text
road<TAB>रो ड
mumbai<TAB>मुं ब ई
```

Pass this to the CLI or set the supplemental lexicon path in the app:

```bash
python naturalness_cli.py align --sources human clone_ref --word-lexicon code_mixed_lexicon.txt
```

Every source is aligned in one `mfa align` call with identical settings. Each (source, speaker)
is a separate MFA speaker, so speaker adaptation never mixes human and machine audio.
Alignment output reports requested/aligned files, already-aligned files, failures, and unresolved
words. The app also shows file and word coverage. Evaluation refuses to silently omit a selected
system that has no valid data on the requested split.

**Default: this project's own MFA step.** Mayur's pipeline details are not available, so `align`
above is the pipeline of record. It is self-contained (IndicMFA Hindi acoustic model, word lexicon
built from the model's phone set, identical settings for every source).

**If TextGrids from another aligner arrive later** (for example Mayur's output), import them with
validation instead of copying by hand:

```bash
python naturalness_cli.py import-alignments --source human --folder external/   # external/<speaker>/<sentence_id>.TextGrid
```

Each grid must have a `words` tier whose word sequence equals the stimulus after normalization;
others are rejected and listed. Use one aligner for **every** source you compare (human and each
TTS system). Human durations from one aligner and machine durations from another are not comparable,
so either import his grids for all sources or use none of them.

## 3. Generate machine speech (stock and cloned)

Which TTS to use: **Maya 2 Native** (`maya2-native`, local, Linux + CUDA 12.8) is the main system
for the timing study. Generation is seeded so reruns are repeatable, and the voice is fixed by
`MAYA_VOICE` (default `Ahana`). **Bulbul v3** is the cloud fallback. **MMS-TTS Hindi** is the only
one whose duration weights this project can fine-tune (section 5). The correction layer (section 4)
works on top of any of them. All six ASR models remain a separate benchmark (Research / Batch
Research modes) and never touch these stimuli.

```bash
python naturalness_cli.py generate --system bulbul-v3        # needs SARVAM_API_KEY
python naturalness_cli.py prompt --speaker S01               # 6-12 s prompt cut from the aligned passage
python naturalness_cli.py generate --system clone-indicf5 --speaker-consent  # or clone-xtts
python naturalness_cli.py align --sources bulbul-v3 clone-indicf5
```

Cloning conditions **only** on `clone_ref/` (the prompt for IndicF5, the passage for XTTS), and
`generate` refuses to run if the corpus check reports leakage. The app and CLI also require
confirmation that every cloned speaker gave explicit permission. The passage must be newly
recorded, must not contain experimental stimuli, and must only be used as the cloning reference.
Review each backend's current terms before installation or use.

Cloning models conflict with the app's pinned packages: IndicF5 needs **Python 3.10 and
transformers 4.50**, while the app runs Python 3.11 with transformers 5.6.2 (for Maya). So IndicF5
lives in its own environment and the app calls it through `VOICE_CLONE_PYTHON`. The pins are in
`requirements-clone.txt` (torch/torchaudio 2.8.0, torchcodec 0.7, IndicF5 at a fixed commit).

**Local Linux / WSL2:**

```bash
bash setup_clone_env.sh $PWD/.venv-clone      # prints the export line below
export VOICE_CLONE_PYTHON=$PWD/.venv-clone/bin/python
```

**Voice cloning on Hugging Face Spaces (Docker):** add these lines to the `Dockerfile` *before* any
`USER` line (writing to `/opt` needs root), then rebuild:

```dockerfile
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
COPY requirements-clone.txt setup_clone_env.sh /opt/clone-setup/
RUN bash /opt/clone-setup/setup_clone_env.sh /opt/clone-env
ENV VOICE_CLONE_PYTHON=/opt/clone-env/bin/python
```

Also accept the terms of `ai4bharat/IndicF5` on Hugging Face with the account that owns `HF_TOKEN`
(the model is gated), and keep `HF_TOKEN` as a Space secret. The build log should end with
`clone env OK: transformers 4.50.0 | torch 2.8.0 …`. The image grows by several GB.

IndicF5 may be gated on Hugging Face (accept its terms, set `HF_TOKEN`). Its model snapshot is
pinned in `voice_clone.py` for repeatable generation; update that revision deliberately when
changing model versions. XTTS-v2 is under the non-commercial Coqui Public Model License; set
`COQUI_TOS_AGREED=1` once you have read it.
Check each model's licence before use. Both need a GPU for reasonable speed.

## 4. The diff and the correction layer

```bash
python naturalness_cli.py durations
python naturalness_cli.py train --system bulbul-v3
python naturalness_cli.py correct --system bulbul-v3                 # writes bulbul-v3+dur
python naturalness_cli.py align --sources bulbul-v3+dur
python naturalness_cli.py durations
```

**Unit of measurement: prosodic words.** Durations are measured on a word plus the enclitics that
follow it (postpositions का की के को से में पर ने तक, copula/auxiliaries है हैं था थी थे थीं हो, particles
भी ही सा सी), e.g. `पानी की`, `जाती है`. The grouping comes from the text, so it is identical for the
human and every system. On the first pilot, MFA squeezed 40 of 57 rejected words, almost all such
enclitics, to 10–20 ms, while the outer edges of the pair were reliable. A unit of k words may last
1.5 s + 0.5 s × (k − 1) before it counts as "long".

**Check alignment quality before trusting any number.** The coverage table reports
`Collapsed (<25 ms/akshara) %`: words the aligner squeezed to almost nothing. Above a few percent,
the duration differences mostly measure aligner error. **Results → Show alignment inspector**
shows the waveform, spectrogram and boundaries for any sentence. The exact MFA lexicon used is
saved as `alignments/_lexicon_used.dict` (phone set: `_lexicon_used.phones.txt`).

`pairs.csv` holds, for every matched unit: `diff_s = human − machine` (the error your professor
described), `log_ratio = log(human/machine)`, `log_ratio_resid` (with each sentence's overall
speaking rate removed) and `pause_diff_s`. Sentences with any badly aligned word (OOV, under
30 ms, over 1.5 s, or a word sequence that differs from the text) are dropped whole.

**What "feeding the diff back" means here.** Bulbul is an API with only a global `pace` setting,
so there are no duration weights to fine-tune. The diff instead trains a correction layer that
sits on top of any TTS:

- **Target:** `log(human/machine)` for each word, plus the human pause after each word.
- **Features:** machine timing, akshara/syllable/matra counts, conjuncts, function word,
  position in the sentence, following punctuation, question or statement, and the speaker's
  average rate (learned on train only).
- **Candidates:** **no correction** (the baseline every layer must beat), a global rate factor,
  per-bucket means, ridge regression and gradient boosting. Pauses are either kept as the machine
  produced them, or learned with a "pause or not" classifier plus a pause-length regressor (where
  no pause is predicted, the machine's short gap is kept or shrunk to the typical human
  between-word gap, never deleted). If "no correction" wins on dev, `train` says so and `correct`
  leaves the audio unchanged. Ties go to the simpler model. Every candidate is fitted on
  **train**, the best is chosen on **dev**, and the chosen one is refitted on train + dev.
  **Test is never used.**
- **Applying it:** each word is time-stretched to its predicted duration with Praat
  overlap-add, which keeps the pitch unchanged (stretch factors clamped to 0.6–1.6×).
  Pauses are shortened or lengthened.
- **The result is measured, not assumed:** corrected audio is aligned again with MFA before
  evaluation.

## 5. Fine-tune the MMS-TTS Hindi duration predictor

The app now has an experimental **weight fine-tuning** path for MMS-TTS Hindi. It optimizes the
checkpoint's stochastic duration predictor with MMS's native duration negative-log-likelihood
objective. The text encoder, posterior encoder, flow, and waveform decoder remain frozen; this is
not full end-to-end VITS training. The Transformers `VitsModel` API does not currently implement
end-to-end training.

The training targets use `dur_s_m + diff_s` (the measured machine duration plus the
human-minus-machine error, i.e. the human duration) for each valid word, and the human pause for
each inter-word space. Within a word, frames are shared across its tokens **in proportion to the
base MMS model's own predicted token durations**. This includes the blank token VITS inserts after
every character: giving blanks zero frames would teach a timing distribution the acoustic model
never saw. Sentences that cannot be mapped (for example, a character missing from the MMS
vocabulary) are skipped and listed in the metadata rather than aborting the run. This is
word-level supervision, not phoneme-level supervision. Inspect alignments and use
enough varied training sentences before drawing conclusions.

First generate and align the baseline MMS audio as well as human recordings, then rebuild
`pairs.csv`:

```bash
python naturalness_cli.py generate --system mms-tts-hin
python naturalness_cli.py align --sources human mms-tts-hin
python naturalness_cli.py durations
python naturalness_cli.py mms-finetune --epochs 50 --learning-rate 0.00001 --device cuda
python naturalness_cli.py generate --system mms-tts-hin-ft --splits dev test --device cuda
python naturalness_cli.py align --sources mms-tts-hin-ft
python naturalness_cli.py durations
python naturalness_cli.py evaluate --systems mms-tts-hin mms-tts-hin-ft --acoustic
```

The fine-tuner trains on `train`, selects the checkpoint using `dev`, and never uses `test`
examples. It requires at least five valid training sentences and two valid dev sentences; these
are technical minimums, not a recommended research sample size. Use substantially more varied,
consented speech for a meaningful experiment. The
duration-predictor checkpoint and metadata are saved under `models/`. The fine-tuned system is
generated separately as `mms-tts-hin-ft`; the stock `mms-tts-hin` output remains unchanged.
Fine-tuning needs PyTorch, Transformers, and the same pinned MMS checkpoint as the app. CPU is
supported but slower; a CUDA device is recommended.

**License:** Meta's [`facebook/mms-tts-hin`](https://huggingface.co/facebook/mms-tts-hin) model
card specifies **CC-BY-NC 4.0**. Fine-tuning does not change those restrictions. Do not use this
model or derived checkpoints commercially unless the rights holder grants permission.

## 6. Evaluation

```bash
python naturalness_cli.py evaluate --systems bulbul-v3 clone-indicf5 bulbul-v3+dur clone-indicf5+dur clone-indicf5+oracle --acoustic
```

All systems are scored on the **same** test sentences: any sentence missing for one system is
dropped for all of them. The run writes `eval/report_test.csv` and `eval/words_test.csv`.

| Metric | Meaning | Better |
|---|---|---|
| `word_rmse_ms`, `word_mae_ms` (95% CI by sentence) | word-duration error vs the human | lower |
| `mean_diff_ms (human-machine)` | positive = machine too fast | near 0 |
| `rhythm_corr` (95% CI) | word-timing pattern with overall rate removed | higher |
| `pause_rmse_ms`, `pause_f1` | pause length / placement (pause = gap ≥ 100 ms) | lower / higher |
| `speech_span_ratio` | total speech duration, machine ÷ human | near 1 |
| `word_f0_corr` | pitch per word vs the human | higher |
| `speaker_sim` | WavLM speaker-verification cosine with the real speaker | higher |
| `utmos` | predicted MOS | higher |
| `synthetic_prob` | zero-shot deepfake detector's score | lower |

Use the 2×2 ablation (generic vs cloned voice, with vs without `+dur`) as the main table. Report
`+oracle` only as the ceiling.

### Listening test (human vs machine)

```bash
python naturalness_cli.py listening-kit --systems bulbul-v3 clone-indicf5 clone-indicf5+dur
```

Then open the app → **Naturalness** → **Listening test**. Each listener enters an ID and gives
consent. Each trial plays the human and one system on the same test sentence (A/B order is
random), and the listener picks the real person.

- Clip files are named `<trial>_A.wav` / `<trial>_B.wav` only, so a copied kit does not reveal
  which clip is human.
- All clips are resampled to 16 kHz, trimmed to 150 ms around the speech and loudness-matched,
  so only the voice gives the answer away.
- Each listener hears each sentence only once, under one system (Latin square). The `+oracle`
  system is never included.
- Aim for 10–20 listeners.

```bash
python naturalness_cli.py listening-report
```

The report shows how often listeners found the human: **50% = indistinguishable**. It gives a
Wilson 95% CI and an exact binomial test against 50%. A wide CI that includes 50% means "not
distinguishable at this sample size", not proof of equivalence, so collect more trials before
claiming it. Spaces storage is not persistent, so download `responses.csv` after each session.

## Tests

```bash
python tests/test_pipeline.py      # end-to-end on synthetic speech, fake MFA, no GPU or network
python tests/test_naturalness.py
```

`test_pipeline.py` checks each action item: durations and `diff_s = human − machine`, a layer that
beats "no correction" on dev and never sees test, corrected audio closer to the human on test
(word RMSE and pause placement), cloning blocked by passage or prompt leakage and by missing
consent, blind listening-clip names, validated TextGrid import, and MMS token targets that sum to
the human frames with blanks included.

This runs the whole pipeline on synthetic speech, with `tests/fake_mfa.py` standing in for MFA.
It also covers supplemental lexicons/code-mixed OOV reporting against the real Hindi acoustic
model, TextGrid parsing, split validation, recording import, cloning consent, selected-system
evaluation completeness, and leakage checks. It does not validate real-world speech alignment
quality or replace manual inspection of a pilot.
