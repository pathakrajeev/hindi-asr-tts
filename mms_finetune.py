"""MMS-TTS Hindi duration-predictor fine-tuning from MFA-aligned human/TTS pairs.

The Transformers VitsModel API does not expose end-to-end VITS training. This module therefore
freezes the text encoder, posterior encoder, flow, and waveform decoder, and fine-tunes only MMS's
stochastic duration predictor with its native negative-log-likelihood objective.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import naturalness as nat

BASE_SYSTEM = "mms-tts-hin"
FINETUNED_SYSTEM = "mms-tts-hin-ft"
BASE_MODEL_ID = "facebook/mms-tts-hin"
BASE_MODEL_REVISION = "1d83b223ec78e30b944f7d96bd117eb3d7023303"
MIN_TRAIN_SENTENCES = 5
MIN_DEV_SENTENCES = 2
_PUNCTUATION = {"comma": ",", "question": "?", "final": "।"}
_INFERENCE_MODELS: dict[tuple[str, str], tuple[object, object]] = {}


@dataclass(frozen=True)
class DurationExample:
    sentence_id: str
    split: str
    input_ids: list[int]
    duration_frames: list[float]


def checkpoint_path(corpus: nat.Corpus) -> Path:
    return corpus.root / "models" / "mms-tts-hin-duration.pt"


def _allocate_frames(duration_s: float, chars: int, frames_per_second: float) -> list[float]:
    if not math.isfinite(duration_s) or duration_s <= 0 or chars <= 0:
        raise ValueError("Word durations and token counts must be positive and finite.")
    total = max(0, int(round(duration_s * frames_per_second)))
    base, remainder = divmod(total, chars)
    return [float(base + (index < remainder)) for index in range(chars)]


def _distribute(total: int, weights: list[float]) -> list[float]:
    """Split an integer frame count over tokens in proportion to `weights` (largest remainder).
    Equal shares when every weight is zero."""
    if not weights:
        return []
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
    raw = w * total
    out = np.floor(raw)
    for index in np.argsort(-(raw - out), kind="stable")[: int(total - out.sum())]:
        out[index] += 1
    return out.tolist()


def build_duration_example(
    text: str,
    rows: pd.DataFrame,
    tokenizer,
    frames_per_second: float,
    base_token_frames=None,
) -> DurationExample:
    """Map MFA word durations and pauses to MMS token duration targets.

    Each word gets the human duration (machine duration + diff), each inter-word space the human
    pause, punctuation markers zero. Within a unit (word / space / marker), frames are shared over
    its tokens in proportion to the BASE model's own predicted token durations when
    `base_token_frames(input_ids)` is given. VITS inserts a blank token after every character; the
    pretrained predictor gives blanks a real share of time, so assigning them zero (or splitting
    evenly) would teach a timing distribution the acoustic model never saw. Each blank belongs to
    the character before it (the leading blank to the first character).
    """
    if rows.empty:
        raise ValueError("MFA duration rows are empty.")
    if not math.isfinite(frames_per_second) or frames_per_second <= 0:
        raise ValueError("The MMS frame rate must be positive and finite.")
    sentence = str(rows.sentence_id.iloc[0])
    expected = nat.unit_tokens(text)   # prosodic units, as in durations.csv / pairs.csv
    rows = rows.sort_values("word_idx").reset_index(drop=True)
    if len(rows) != len(expected):
        raise ValueError(f"{sentence}: MFA pair count does not match the stimulus word count.")
    if rows.word.astype(str).tolist() != [word for word, _ in expected]:
        raise ValueError(f"{sentence}: MFA words do not match the normalized stimulus.")

    units: list[tuple[str, int]] = []   # (characters, total frames)
    for index, (word, punctuation) in enumerate(expected):
        if index:
            pause = float(rows.iloc[index - 1].pause_after_s_h)
            if not math.isfinite(pause) or pause < 0:
                raise ValueError(f"{sentence}: pause durations must be finite and non-negative.")
            units.append((" ", max(0, int(round(pause * frames_per_second)))))
        human_s = float(rows.iloc[index].dur_s_m) + float(rows.iloc[index].diff_s)
        if not math.isfinite(human_s) or human_s <= 0:
            raise ValueError(f"{sentence}: word durations must be positive and finite.")
        units.append((word, max(1, int(round(human_s * frames_per_second)))))
        marker = _PUNCTUATION.get(punctuation)
        if marker:
            units.append((marker, 0))

    raw_text = "".join(chars for chars, _ in units)
    char_unit = [u for u, (chars, _) in enumerate(units) for _ in chars]
    model_text = unicodedata.normalize("NFC", raw_text)
    model_text = tokenizer.normalize_text(model_text) if hasattr(tokenizer, "normalize_text") else model_text
    if len(model_text) != len(char_unit):
        raise ValueError(f"{sentence}: tokenizer normalization changes text length; cannot map MFA durations safely.")
    # VitsTokenizer silently drops characters missing from its vocabulary (MMS Hindi has no "।", "?"
    # or ","). Drop them here the same way, so tokens and duration targets stay aligned. A dropped
    # punctuation marker carries 0 frames, so the human timing total is unchanged.
    vocab = set(tokenizer.get_vocab()) if hasattr(tokenizer, "get_vocab") else None
    if vocab:
        kept = [i for i, c in enumerate(model_text) if c in vocab]
        model_text = "".join(model_text[i] for i in kept)
        char_unit = [char_unit[i] for i in kept]
        if not char_unit:
            raise ValueError(f"{sentence}: no characters of this sentence are in the MMS vocabulary.")
        # A unit with frames but no surviving characters (e.g. a pause, if " " were missing) gives its
        # frames to the nearest earlier unit that still has tokens, so the sentence total is kept.
        present = set(char_unit)
        for u, (chars, total) in enumerate(units):
            if total > 0 and u not in present:
                host = max((k for k in present if k < u), default=min(present))
                units[host] = (units[host][0], units[host][1] + total)
                units[u] = (chars, 0)

    encoded = tokenizer(model_text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    add_blank = bool(getattr(tokenizer, "add_blank", False))
    expected_length = len(model_text) * (2 if add_blank else 1) + int(add_blank)
    if len(input_ids) != expected_length:
        raise ValueError(
            f"{sentence}: unexpected MMS tokenization length ({len(input_ids)} tokens for "
            f"{len(model_text)} characters); a character is probably missing from the MMS vocabulary."
        )

    # token -> owning unit
    if add_blank:
        token_unit = [char_unit[0]] + [char_unit[k] for k in range(len(char_unit)) for _ in (0, 1)]
    else:
        token_unit = list(char_unit)
    weights = [1.0] * len(input_ids)
    if base_token_frames is not None:
        weights = [float(v) for v in base_token_frames(input_ids)]
        if len(weights) != len(input_ids):
            raise ValueError(f"{sentence}: base duration prediction length does not match the tokens.")
    token_frames = [0.0] * len(input_ids)
    for u, (_, total) in enumerate(units):
        members = [t for t, owner in enumerate(token_unit) if owner == u]
        for t, frames in zip(members, _distribute(total, [weights[t] for t in members])):
            token_frames[t] = frames
    if sum(token_frames) <= 0:
        raise ValueError(f"{sentence}: no positive duration targets were produced.")
    return DurationExample(
        sentence_id=sentence,
        split=str(rows.split.iloc[0]),
        input_ids=[int(value) for value in input_ids],
        duration_frames=token_frames,
    )


def prepare_examples(
    corpus: nat.Corpus,
    pairs: pd.DataFrame,
    tokenizer,
    frames_per_second: float,
    splits: tuple[str, ...] = ("train", "dev"),
    base_token_frames=None,
    log=print,
) -> tuple[list[DurationExample], list[str]]:
    """Build only the requested split examples; callers train with train/dev, never test.
    A sentence that cannot be mapped is skipped and reported, not allowed to abort the run."""
    texts = corpus.text_of()
    selected = pairs[pairs.split.isin(splits)]
    examples, skipped = [], []
    for (speaker, sentence_id), group in selected.groupby(["speaker", "sentence_id"], sort=True):
        if sentence_id not in texts:
            raise ValueError(f"Unknown sentence ID in duration pairs: {sentence_id}.")
        try:
            examples.append(build_duration_example(texts[sentence_id], group, tokenizer, frames_per_second,
                                                   base_token_frames))
        except ValueError as exc:
            skipped.append(f"{speaker}/{sentence_id}: {exc}")
            log(f"skipped {speaker}/{sentence_id}: {exc}")
    return examples, skipped


def _load_base_model(device: str):
    from transformers import VitsModel, VitsTokenizer

    model = VitsModel.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION).to(device)
    tokenizer = VitsTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION)
    if not model.config.use_stochastic_duration_prediction:
        raise RuntimeError("The selected MMS checkpoint does not use a stochastic duration predictor.")
    if getattr(tokenizer, "is_uroman", False):
        raise RuntimeError("The Hindi MMS tokenizer requires uroman, which is not supported by this trainer.")
    return model, tokenizer


def fine_tune_duration_predictor(
    corpus: nat.Corpus,
    epochs: int = 50,
    learning_rate: float = 1e-5,
    patience: int = 6,
    seed: int = 13,
    device: str = "cpu",
    log=print,
) -> dict:
    """Fine-tune only MMS's stochastic duration predictor from train/dev human-vs-MMS pairs."""
    import torch

    if epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0 or patience < 1:
        raise ValueError("epochs and patience must be positive, and learning_rate must be positive and finite.")
    if not corpus.splits_csv.is_file():
        raise FileNotFoundError("Create fixed train/dev/test splits before fine-tuning MMS-TTS.")
    pairs = nat.load_pairs(corpus, BASE_SYSTEM)
    if pairs.empty:
        raise ValueError("No valid MMS/human pairs. Align both `human` and `mms-tts-hin`, then rebuild durations.")
    pairs = pairs[pairs.split.isin(("train", "dev"))]
    sentence_counts = pairs.groupby("split").sentence_id.nunique().to_dict()
    if sentence_counts.get("train", 0) < MIN_TRAIN_SENTENCES:
        raise ValueError(
            f"Fine-tuning needs at least {MIN_TRAIN_SENTENCES} valid train sentences; "
            f"found {sentence_counts.get('train', 0)}."
        )
    if sentence_counts.get("dev", 0) < MIN_DEV_SENTENCES:
        raise ValueError(
            f"Fine-tuning needs at least {MIN_DEV_SENTENCES} valid dev sentences for checkpoint selection; "
            f"found {sentence_counts.get('dev', 0)}."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, tokenizer = _load_base_model(device)
    sample_rate = int(model.config.sampling_rate)
    hop_length = math.prod(model.config.upsample_rates)
    frames_per_second = sample_rate / hop_length
    predictor = model.duration_predictor

    def base_token_frames(input_ids: list[int]) -> list[float]:
        """The unchanged base model's own token durations (noise off), in frames."""
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        mask = torch.ones_like(ids, dtype=torch.float32)
        with torch.no_grad():
            hidden = model.text_encoder(input_ids=ids, padding_mask=mask.unsqueeze(-1),
                                        attention_mask=mask).last_hidden_state.transpose(1, 2)
            log_w = predictor(hidden, mask.unsqueeze(1), reverse=True, noise_scale=0.0)
        return torch.exp(log_w).reshape(-1).cpu().tolist()

    model.eval()
    examples, skipped = prepare_examples(corpus, pairs, tokenizer, frames_per_second,
                                         base_token_frames=base_token_frames, log=log)
    train_examples = [example for example in examples if example.split == "train"]
    dev_examples = [example for example in examples if example.split == "dev"]
    if len(train_examples) < MIN_TRAIN_SENTENCES or len(dev_examples) < MIN_DEV_SENTENCES:
        raise ValueError(
            f"Only {len(train_examples)} train / {len(dev_examples)} dev sentences could be converted to MMS "
            f"duration targets (need {MIN_TRAIN_SENTENCES}/{MIN_DEV_SENTENCES}). Skipped: {skipped[:5]}"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in predictor.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=learning_rate)

    def nll(example: DurationExample, training: bool) -> torch.Tensor:
        input_ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
        mask = torch.ones_like(input_ids, dtype=torch.float32)
        with torch.set_grad_enabled(training):
            encoded = model.text_encoder(
                input_ids=input_ids,
                padding_mask=mask.unsqueeze(-1),
                attention_mask=mask,
            )
            hidden = encoded.last_hidden_state.transpose(1, 2)
            loss_values = predictor(
                hidden,
                mask.unsqueeze(1),
                durations=torch.tensor(
                    [example.duration_frames],
                    dtype=hidden.dtype,
                    device=device,
                ).unsqueeze(1),
                reverse=False,
            )
            return loss_values.mean()

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.text_encoder.eval()
        predictor.train()
        order = np.random.permutation(len(train_examples))
        train_losses = []
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss = nll(train_examples[int(index)], training=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite MMS duration loss at epoch {epoch}; training stopped.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        predictor.eval()
        with torch.no_grad():
            dev_losses = [float(nll(example, training=False).cpu()) for example in dev_examples]
        dev_loss = float(np.mean(dev_losses))
        train_loss = float(np.mean(train_losses))
        log(f"epoch {epoch}/{epochs}: train NLL={train_loss:.4f}, dev NLL={dev_loss:.4f}")
        if not math.isfinite(dev_loss):
            raise RuntimeError(f"Non-finite MMS dev loss at epoch {epoch}; training stopped.")
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in predictor.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("MMS fine-tuning did not produce a valid checkpoint.")
    predictor.load_state_dict(best_state)
    output = checkpoint_path(corpus)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".pt", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(best_state, temp_path)
        temp_path.replace(output)
    finally:
        temp_path.unlink(missing_ok=True)

    metadata = {
        "base_model": BASE_MODEL_ID,
        "base_revision": BASE_MODEL_REVISION,
        "fine_tuned_component": "stochastic_duration_predictor",
        "frozen_components": ["text_encoder", "posterior_encoder", "flow", "waveform_decoder"],
        "objective": "MMS VITS stochastic duration predictor negative log-likelihood",
        "duration_target": "MMS word duration + (human word duration - MMS word duration)",
        "token_allocation": "per word/pause, proportional to the base model's own token durations (noise 0)",
        "skipped_sentences": skipped,
        "training_split": "train",
        "selection_split": "dev",
        "held_out_split": "test",
        "train_sentences": len(train_examples),
        "dev_sentences": len(dev_examples),
        "train_words": int(sum(len(group) for _, group in pairs[pairs.split == "train"].groupby("sentence_id"))),
        "dev_words": int(sum(len(group) for _, group in pairs[pairs.split == "dev"].groupby("sentence_id"))),
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "frames_per_second": frames_per_second,
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_dev_nll": best_loss,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _INFERENCE_MODELS.clear()
    log(f"Saved fine-tuned MMS duration predictor: {output} (best epoch {best_epoch})")
    return metadata


def _load_finetuned_model(device: str, checkpoint: Path):
    import torch

    key = (str(checkpoint.resolve()), device)
    cached = _INFERENCE_MODELS.get(key)
    if cached:
        return cached
    import research

    base_model, tokenizer = research.load_mms(device)
    model = copy.deepcopy(base_model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.duration_predictor.load_state_dict(state, strict=True)
    model.eval()
    _INFERENCE_MODELS[key] = (model, tokenizer)
    return model, tokenizer


def synthesize(text: str, device: str, checkpoint: Path) -> tuple[np.ndarray, int]:
    """Generate with the fine-tuned duration predictor and the unchanged MMS acoustic components."""
    import torch

    metadata_path = checkpoint.with_suffix(".json")
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("MMS duration fine-tuning checkpoint or metadata is missing.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("base_model") != BASE_MODEL_ID or metadata.get("base_revision") != BASE_MODEL_REVISION:
        raise ValueError("MMS fine-tuning checkpoint does not match the pinned base model revision.")
    model, tokenizer = _load_finetuned_model(device, checkpoint)
    from research import TTS_SEED, sentence_chunks

    sample_rate = int(model.config.sampling_rate)
    silence = np.zeros(int(0.18 * sample_rate), dtype=np.float32)
    parts = []
    torch.manual_seed(TTS_SEED)
    with torch.inference_mode():
        for chunk in sentence_chunks(text, 220):
            inputs = tokenizer(chunk, return_tensors="pt")
            if inputs["input_ids"].shape[-1] == 0:
                continue
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            parts.extend([model(**inputs).waveform[0].float().cpu().numpy(), silence])
    if not parts:
        raise RuntimeError("Fine-tuned MMS-TTS produced no audio for this text.")
    return np.concatenate(parts[:-1]).astype(np.float32), sample_rate
