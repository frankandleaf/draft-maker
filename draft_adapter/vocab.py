"""Cross-tokenizer vocabulary alignment for ordinary CausalLM checkpoints.

The bridge is deliberately offline: a target tokenizer becomes the canonical
vocabulary, while a source/student model keeps its transformer body.  The
result is still a normal Hugging Face CausalLM and does not require a custom
runtime plugin.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


def tokenizer_hash(tokenizer: PreTrainedTokenizerBase) -> str:
    """Return a stable hash of the tokenizer vocabulary and special tokens."""
    items = sorted((str(token), int(index)) for token, index in tokenizer.get_vocab().items())
    payload = {
        "vocab": items,
        "special_tokens": tokenizer.all_special_tokens,
        "special_ids": tokenizer.all_special_ids,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _decode_one(tokenizer: PreTrainedTokenizerBase, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


@dataclass
class VocabularyMapping:
    """Target-token to source-token decomposition table."""

    target_vocab_size: int
    source_vocab_size: int
    target_to_source: list[list[int]]
    exact_source_ids: list[int]
    special_target_ids: list[int]
    fallback_target_ids: list[int]
    source_tokenizer_hash: str
    target_tokenizer_hash: str

    @property
    def exact_count(self) -> int:
        return sum(item >= 0 for item in self.exact_source_ids)

    @property
    def decomposable_count(self) -> int:
        return sum(
            bool(self.target_to_source[index]) and item < 0
            for index, item in enumerate(self.exact_source_ids)
        )

    def stats(self) -> dict[str, Any]:
        lengths = [len(row) for row in self.target_to_source if row]
        return {
            "target_vocab_size": self.target_vocab_size,
            "source_vocab_size": self.source_vocab_size,
            "exact_count": self.exact_count,
            "exact_ratio": self.exact_count / max(self.target_vocab_size, 1),
            "decomposable_count": self.decomposable_count,
            "fallback_count": len(self.fallback_target_ids),
            "fallback_ratio": len(self.fallback_target_ids) / max(self.target_vocab_size, 1),
            "mean_source_tokens_per_target": (
                sum(lengths) / max(len(lengths), 1)
            ),
            "max_source_tokens_per_target": max(lengths, default=0),
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "target_vocab_size": self.target_vocab_size,
            "source_vocab_size": self.source_vocab_size,
            "target_to_source": self.target_to_source,
            "exact_source_ids": self.exact_source_ids,
            "special_target_ids": self.special_target_ids,
            "fallback_target_ids": self.fallback_target_ids,
            "source_tokenizer_hash": self.source_tokenizer_hash,
            "target_tokenizer_hash": self.target_tokenizer_hash,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "VocabularyMapping":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _decoded_index(tokenizer: PreTrainedTokenizerBase) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for token_id in range(len(tokenizer)):
        text = _decode_one(tokenizer, token_id)
        index.setdefault(text, []).append(token_id)
    return index


def build_vocabulary_mapping(
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
    max_decomposition: int = 32,
) -> VocabularyMapping:
    """Build an offline target-vocabulary -> source-vocabulary mapping.

    Exact decoded strings form the fast path.  Other target tokens are
    decomposed with the source tokenizer.  A row is marked fallback when the
    source tokenizer cannot reproduce the target token text exactly; training
    will then learn that row from teacher data.
    """
    source_index = _decoded_index(source_tokenizer)
    source_special_ids = set(int(item) for item in source_tokenizer.all_special_ids)
    target_special_ids = set(int(item) for item in target_tokenizer.all_special_ids)
    target_to_source: list[list[int]] = []
    exact_source_ids: list[int] = []
    fallback: list[int] = []

    for target_id in range(len(target_tokenizer)):
        text = _decode_one(target_tokenizer, target_id)
        candidates = [
            item for item in source_index.get(text, [])
            if item not in source_special_ids
        ]
        if target_id in target_special_ids:
            # Special tokens are only exact when their literal token exists on
            # the source side; otherwise they receive a trainable new row.
            literal = target_tokenizer.convert_ids_to_tokens(target_id)
            source_literal = source_tokenizer.get_vocab().get(literal)
            candidates = [int(source_literal)] if source_literal is not None else []
            if not candidates:
                # Never decompose a control token into ordinary text pieces.
                # Its row must be learned with the target chat template.
                target_to_source.append([])
                exact_source_ids.append(-1)
                fallback.append(target_id)
                continue

        if candidates:
            source_id = int(candidates[0])
            target_to_source.append([source_id])
            exact_source_ids.append(source_id)
            continue

        pieces = source_tokenizer(
            text,
            add_special_tokens=False,
        ).get("input_ids", [])
        if pieces and len(pieces) <= max_decomposition:
            reconstructed = source_tokenizer.decode(
                pieces,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if reconstructed == text and not any(
                int(piece) in source_special_ids for piece in pieces
            ):
                target_to_source.append([int(piece) for piece in pieces])
                exact_source_ids.append(-1)
                continue

        target_to_source.append([])
        exact_source_ids.append(-1)
        fallback.append(target_id)

    return VocabularyMapping(
        target_vocab_size=len(target_tokenizer),
        source_vocab_size=len(source_tokenizer),
        target_to_source=target_to_source,
        exact_source_ids=exact_source_ids,
        special_target_ids=sorted(target_special_ids),
        fallback_target_ids=fallback,
        source_tokenizer_hash=tokenizer_hash(source_tokenizer),
        target_tokenizer_hash=tokenizer_hash(target_tokenizer),
    )


@torch.no_grad()
def initialize_model_for_target_vocab(
    model: torch.nn.Module,
    mapping: VocabularyMapping,
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
    seed: int = 42,
) -> dict[str, int | float]:
    """Resize and initialize a source model to the target tokenizer."""
    if len(source_tokenizer) != mapping.source_vocab_size:
        raise ValueError("source tokenizer does not match the mapping")
    if len(target_tokenizer) != mapping.target_vocab_size:
        raise ValueError("target tokenizer does not match the mapping")

    old_input = model.get_input_embeddings().weight.detach().float().clone()
    old_output_module = model.get_output_embeddings()
    old_output = (
        old_output_module.weight.detach().float().clone()
        if old_output_module is not None
        else old_input
    )
    old_vocab = old_input.shape[0]
    if old_vocab < mapping.source_vocab_size:
        raise ValueError(
            f"model embedding has {old_vocab} rows, "
            f"but tokenizer has {mapping.source_vocab_size}"
        )

    model.resize_token_embeddings(mapping.target_vocab_size)
    new_input = model.get_input_embeddings().weight
    new_output_module = model.get_output_embeddings()
    new_output = new_output_module.weight if new_output_module is not None else new_input

    generator = torch.Generator(device="cpu").manual_seed(seed)
    std = float(old_input.std().item())
    new_input.data.normal_(mean=0.0, std=max(std, 1e-3), generator=generator)
    if new_output.data_ptr() != new_input.data_ptr():
        new_output.data.normal_(mean=0.0, std=max(float(old_output.std().item()), 1e-3), generator=generator)

    exact = decomposed = fallback = 0
    for target_id, source_ids in enumerate(mapping.target_to_source):
        if not source_ids:
            fallback += 1
            continue
        indices = torch.tensor(source_ids, dtype=torch.long, device=old_input.device)
        weights = torch.ones(len(source_ids), dtype=old_input.dtype, device=old_input.device)
        weights /= weights.sum()
        new_input.data[target_id].copy_((old_input[indices] * weights[:, None]).sum(0).to(new_input.dtype))
        if new_output.data_ptr() != new_input.data_ptr():
            new_output.data[target_id].copy_((old_output[indices] * weights[:, None]).sum(0).to(new_output.dtype))
        if len(source_ids) == 1:
            exact += 1
        else:
            decomposed += 1

    model.config.vocab_size = mapping.target_vocab_size
    if hasattr(model.config, "text_config") and model.config.text_config is not None:
        model.config.text_config.vocab_size = mapping.target_vocab_size
    model.config.bos_token_id = target_tokenizer.bos_token_id
    model.config.eos_token_id = target_tokenizer.eos_token_id
    model.config.pad_token_id = target_tokenizer.pad_token_id
    return {
        "exact_initialized": exact,
        "decomposed_initialized": decomposed,
        "fallback_random": fallback,
        "target_vocab_size": mapping.target_vocab_size,
    }
