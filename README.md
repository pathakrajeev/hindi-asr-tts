# Hindi ASR → TTS Pipeline

A Streamlit app for testing Hindi speech recognition (ASR) and speech synthesis (TTS), and for
research into making synthetic Hindi speech sound more human, mainly in its word timing.

It does three things:

1. **Compares models.** Runs several Hindi ASR and TTS models on the same recording and reports
   WER/CER, latency, prosody, voice measurements and a synthetic-speech (deepfake) score.
2. **Runs a live demo.** Record or upload Hindi audio, transcribe it, edit the transcript and
   speak it back.
3. **Runs the naturalness pipeline.** Aligns human and machine speech word by word with the
   Montreal Forced Aligner (MFA), measures the timing difference (human − machine), uses it to
   correct or fine-tune TTS timing, clones the speaker's voice, and evaluates the result on a
   held-out test set and in a blind listening test.

## Models

| Type | Model | Runs | Needs |
|---|---|---|---|
| ASR | Koyal Hindi 120M | local, ~0.5 GB | `HF_TOKEN` (gated model) |
| ASR | Sarvam Saaras v3 | API | `SARVAM_API_KEY` |
| ASR | Whisper Large-v3-Turbo | local, ~3.2 GB | |
| ASR | IndicConformer 600M (CTC and RNNT) | local, ~2.5 GB | |
| ASR | SraVaani-1.0 | local, ~2–3 GB | |
| TTS | Sarvam Bulbul v3 | API | `SARVAM_API_KEY` |
| TTS | Maya 2 Native | local, Linux + CUDA 12.8, ~9.2 GB VRAM peak | see [MAYA_LOCAL_SETUP.md](MAYA_LOCAL_SETUP.md) |
| TTS | Meta MMS-TTS Hindi | local, ~0.15 GB | licence CC-BY-NC 4.0 |
| Voice clone | IndicF5 (or XTTS-v2) | separate Python 3.10 environment, GPU | `VOICE_CLONE_PYTHON`, `HF_TOKEN` |

Local models load only when you select them.

## The three modes in the app

Choose a mode with the selector at the top of the page.

### Research

Compare models on one recording (up to 120 seconds).

1. Enter the correct (reference) transcript, then record or upload the audio.
2. Choose the language variety (Hindi, Hindi-English code-mixed or Sanskrit) and the ASR and TTS
   models to compare. Model and language combinations that a model does not support are listed
   as unsupported and skipped.
3. Choose what the TTS models read: the reference text, an ASR transcript, or your own text.
4. Review the results tabs: preprocessing and voice-activity checks, WER/CER (normalized, raw and
   script-tolerant) with word-level alignment, latency and real-time factor, TTS loopback and
   UTMOS scores, prosody and voice measurements, and an optional synthetic-speech check.

### Live

A quick end-to-end demo. It needs `HF_TOKEN` and `SARVAM_API_KEY`.

1. Record or upload Hindi audio (up to 25 minutes, 50 MB).
2. Koyal transcribes it. Edit the transcript if needed.
3. Confirm the text. Bulbul v3 then speaks it.
4. Review the generated audio, recording diagnostics, pitch and energy contours and timing.

Live mode has no reference transcript, so it does not report WER or CER.

### Batch Research

The full research workflow, in five steps on the page:

1. **Upload**: test recordings with their transcripts, plus a separate 2–5 minute read-aloud
   passage that is used only for voice cloning.
2. **Benchmark**: corpus WER/CER with 95% confidence intervals, TTS scores and a deepfake score
   per clip.
3. **MFA durations**: word durations for human and machine speech, and the difference
   (human − machine).
4. **Fine-tune and clone**: the difference fine-tunes the MMS duration predictor or trains a
   timing-correction layer for the cloned voice.
5. **Evaluate**: scores on the held-out test sentences only (timing error, speaker similarity,
   deepfake score) and a blind listening test where 50% correct means listeners cannot tell
   human from machine.

[TESTING.md](TESTING.md) is a click-by-click checklist for this mode with the expected result of
each step.

## Getting started

### Option 1: Docker (recommended)

The Docker image installs everything: MFA, ffmpeg, the app's Python environment and the separate
voice-cloning environment. The image is large (several GB).

```bash
docker build -t hindi-asr-tts .
docker run --gpus all -p 7860:7860 \
  -e HF_TOKEN=hf_xxx \
  -e SARVAM_API_KEY=xxx \
  -v "$PWD/data:/data" \
  hindi-asr-tts
```

Open http://localhost:7860. Leave out `--gpus all` on a machine without an NVIDIA GPU. Maya and
voice cloning then won't run, but the other models will.

The `-v` mount keeps recordings, splits and trained models in `./data` when the container stops.

### Option 2: Hugging Face Spaces

This repository is set up as a Docker Space (see the front matter at the top of this file).

1. Create a Docker Space and push this repository to it.
2. Add `HF_TOKEN` and `SARVAM_API_KEY` as Space **secrets**.
3. On the Hugging Face website, accept the terms of `ai4bharat/IndicF5` (and Koyal, which is
   also gated) with the account that owns `HF_TOKEN`.
4. Choose GPU hardware if you need Maya, voice cloning or fast fine-tuning.
5. Attach a **read-write storage volume at `/data`**. Space disk is wiped on restart, so without
   it you lose recordings and trained models. If you mount it somewhere else, set
   `NATURALNESS_ROOT` to a writable folder inside it.

### Option 3: Run locally without Docker

You need Linux or WSL2, Python 3.11, ffmpeg and, for Maya, an NVIDIA GPU with CUDA 12.8. Native
Windows can run the app but not Maya.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# MFA (needed for the naturalness pipeline), in its own conda environment:
conda create -n mfa -c conda-forge python=3.11 montreal-forced-aligner
# then put that environment's bin folder on PATH

# Optional: the voice-cloning environment (Python 3.10, kept separate because of package conflicts)
bash setup_clone_env.sh $PWD/.venv-clone
export VOICE_CLONE_PYTHON=$PWD/.venv-clone/bin/python

streamlit run app.py
```

Then open http://localhost:8501.

### Secrets and settings

Put these in a `.env` file in the project folder (it is git-ignored and loaded on start), or set
them as environment variables.

| Variable | What it is for |
|---|---|
| `HF_TOKEN` | Download gated Hugging Face models (Koyal, IndicF5) |
| `SARVAM_API_KEY` | Sarvam Saaras (ASR) and Bulbul (TTS) |
| `MAYA_VOICE` | Maya voice used by the naturalness pipeline (default `Ahana`) |
| `VOICE_CLONE_PYTHON` | Python interpreter of the voice-cloning environment |
| `NATURALNESS_ROOT` | Where the naturalness corpus is stored (default `data/naturalness`; `/data/naturalness` in Docker) |
| `INDICMFA_HI_ACOUSTIC`, `INDICMFA_HI_DICTIONARY` | Use a different MFA acoustic model or dictionary than the bundled ones |
| `COQUI_TOS_AGREED` | Set to `1` after reading the XTTS-v2 licence, if you use XTTS |

Example `.env`:

```env
HF_TOKEN=hf_xxx
SARVAM_API_KEY=xxx
```

## Naturalness pipeline from the command line

Everything in Batch Research steps 3–5 can also be run with `naturalness_cli.py`. A typical run
with one speaker:

```bash
python naturalness_cli.py init --stimuli sentences.txt --speakers S01   # one sentence per line
python naturalness_cli.py split                                         # fixed 70/15/15 split
# record speakers/S01/stimuli/*.wav and speakers/S01/clone_ref/passage.wav + passage.txt
python naturalness_cli.py check                                         # recordings and leakage check

python naturalness_cli.py generate --system bulbul-v3
python naturalness_cli.py align --sources human bulbul-v3
python naturalness_cli.py durations                                     # writes pairs.csv (human − machine)
python naturalness_cli.py train --system bulbul-v3                      # correction layer: fit on train, pick on dev
python naturalness_cli.py correct --system bulbul-v3                    # writes bulbul-v3+dur
python naturalness_cli.py align --sources bulbul-v3+dur
python naturalness_cli.py durations
python naturalness_cli.py evaluate --systems bulbul-v3 bulbul-v3+dur --acoustic

python naturalness_cli.py listening-kit --systems bulbul-v3 bulbul-v3+dur
python naturalness_cli.py listening-report
```

Run `python naturalness_cli.py --help` for all commands, including `prompt` and
`generate --system clone-indicf5` (voice cloning), `mms-finetune` (MMS weight fine-tuning),
`import-alignments` and `convert-audio`.

[NATURALNESS.md](NATURALNESS.md) explains each step in detail: recording rules, the MFA lexicon,
how the correction layer works, MMS fine-tuning, every evaluation metric and the listening test.

### Rules the pipeline enforces

- The train/dev/test split is fixed before any analysis and cannot be changed afterwards.
- Models are trained on train and chosen on dev. Test sentences are used only for the final
  evaluation.
- Voice cloning uses only the separate read-aloud passage. It is blocked if the passage shares
  text or audio with a test sentence, and until you confirm that every speaker gave permission.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit entry point and Live mode |
| `research.py` | Research mode: model registry, inference and result tabs |
| `batch_research.py` | Batch Research upload and benchmark steps |
| `naturalness.py` | Naturalness pipeline logic (alignment, durations, correction, evaluation, listening test) |
| `naturalness_ui.py` | Batch Research steps 3–5 in the app |
| `naturalness_cli.py` | Command-line interface for the naturalness pipeline |
| `mms_finetune.py` | MMS-TTS Hindi duration-predictor fine-tuning |
| `voice_clone.py` | Voice-cloning backends, run in the separate clone environment |
| `deepfake.py` | Synthetic-speech detection |
| `speech_core.py` | Shared audio loading, preprocessing and metrics |
| `textnorm.py`, `stats.py` | Hindi text normalization and WER; bootstrap confidence intervals |
| `Hindi_All_Acoustic.zip`, `Hindi_Dict_g2g.txt` | IndicMFA Hindi acoustic model and akshara dictionary |
| `Sanskrit_Acoustic_Model.zip`, `Sanskrit_Dict.txt` | Sanskrit equivalents |
| `requirements.txt` | App dependencies (Python 3.11) |
| `requirements-clone.txt`, `setup_clone_env.sh` | Voice-cloning environment (Python 3.10) |
| `test_*.py`, `fake_mfa*.py` | Tests and a fake MFA used by them |

## Tests

The tests run the whole naturalness pipeline on synthetic speech with a fake MFA, so they need no
GPU, network or real recordings. `test_m4a.py` needs ffmpeg.

The test scripts expect to be in a `tests/` subfolder (they look one folder up for the app
modules). Move them there before running:

```bash
mkdir -p tests && mv test_*.py fake_mfa.py fake_mfa_cli.py tests/
python tests/test_pipeline.py
python tests/test_naturalness.py
python tests/test_flags.py
python tests/test_m4a.py
```

## Licences and consent

- `facebook/mms-tts-hin` is CC-BY-NC 4.0 (non-commercial). Checkpoints fine-tuned from it carry
  the same restriction.
- XTTS-v2 is under the non-commercial Coqui Public Model License.
- IndicF5 and Koyal are gated on Hugging Face. Accept their terms before use.
- Only clone the voice of someone who has given explicit permission. Speaker recordings are kept
  in `data/`, which is git-ignored. Do not commit recordings or model checkpoints.
