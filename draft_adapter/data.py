"""Public and local text data loading for calibration/distillation."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

from datasets import load_dataset
from torch import Tensor


KNOWN_SPLITS = {"train", "validation", "valid", "test", "dev"}

DATA_PRESETS: dict[str, tuple[str, ...]] = {
    # Stable, broadly useful default. Sources are mixed by quota.
    "public-mixed": (
        "allenai/c4:zh:train",
        "allenai/c4:en:train",
        "Salesforce/wikitext:wikitext-2-raw-v1:train",
    ),
    # Chinese-first calibration. Instruction datasets are best-effort and may
    # fail if the HF dataset is renamed or gated; C4/wiki remain fallback.
    "public-zh": (
        "allenai/c4:zh:train",
        "BelleGroup/train_0.5M_CN:train",
        "YeungNLP/firefly-train-1.1M:train",
        "Salesforce/wikitext:wikitext-2-raw-v1:train",
    ),
    "public-zh-fast": (
        "BelleGroup/train_0.5M_CN:train",
        "YeungNLP/firefly-train-1.1M:train",
        "Salesforce/wikitext:wikitext-2-raw-v1:train",
    ),
    "public-en": (
        "allenai/c4:en:train",
        "Salesforce/wikitext:wikitext-2-raw-v1:train",
    ),
}


@dataclass(frozen=True)
class DataSource:
    """A local file or Hugging Face dataset source."""

    raw: str
    path: str
    config: str | None = None
    split: str = "train"
    local: bool = False

    @property
    def label(self) -> str:
        if self.local:
            return self.path
        if self.config:
            return f"{self.path}:{self.config}:{self.split}"
        return f"{self.path}:{self.split}"


def parse_data_sources(
    data_sources: str | Sequence[str] | None = None,
    preset: str = "public-mixed",
) -> list[DataSource]:
    """Parse local paths or HF dataset specs.

    HF specs use one of:
        dataset
        dataset:split
        dataset:config:split

    Examples:
        allenai/c4:zh:train
        Salesforce/wikitext:wikitext-2-raw-v1:train
        ./calib.jsonl
    """
    if data_sources is None:
        try:
            raw_sources: Sequence[str] = DATA_PRESETS[preset]
        except KeyError as exc:
            valid = ", ".join(sorted(DATA_PRESETS))
            raise ValueError(f"Unknown data preset '{preset}'. Valid presets: {valid}") from exc
    elif isinstance(data_sources, str):
        raw_sources = [s.strip() for s in data_sources.split(",") if s.strip()]
    else:
        raw_sources = [str(s).strip() for s in data_sources if str(s).strip()]

    if not raw_sources:
        raise ValueError("At least one data source is required")
    return [_parse_source(raw) for raw in raw_sources]


def load_calibration_data(
    tokenizer,
    num_samples: int = 16,
    seq_len: int = 512,
    device: str = "cuda",
    data_sources: str | Sequence[str] | None = None,
    data_preset: str = "public-mixed",
    source_timeout: int = 30,
) -> Tensor:
    """Load text data, tokenize it, and pack fixed-length token sequences.

    Public HF datasets are streamed. Local .jsonl/.json/.txt files are read
    directly. Multiple sources are mixed by quota, so calibration does not come
    entirely from whichever dataset responds first.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    sources = parse_data_sources(data_sources, preset=data_preset)
    print("  Data pipeline: " + ", ".join(source.label for source in sources), flush=True)

    samples: list[Tensor] = []
    loaded_by_source: dict[str, int] = {source.raw: 0 for source in sources}

    per_source = max(1, math.ceil(num_samples / len(sources)))
    for source in sources:
        remaining = num_samples - len(samples)
        if remaining <= 0:
            break
        target = min(per_source, remaining)
        new_samples = _load_from_source(
            source, tokenizer, target=target, seq_len=seq_len,
            skip_sequences=loaded_by_source[source.raw],
            timeout=source_timeout,
        )
        loaded_by_source[source.raw] += len(new_samples)
        samples.extend(new_samples)

    # Fill shortfalls without duplicating already returned sequences from a
    # source. This keeps public-mixed useful even if later sources fail.
    if len(samples) < num_samples:
        for source in sources:
            if loaded_by_source[source.raw] == 0:
                continue
            remaining = num_samples - len(samples)
            if remaining <= 0:
                break
            new_samples = _load_from_source(
                source, tokenizer, target=remaining, seq_len=seq_len,
                skip_sequences=loaded_by_source[source.raw],
                timeout=source_timeout,
            )
            loaded_by_source[source.raw] += len(new_samples)
            samples.extend(new_samples)

    if not samples:
        print("  WARNING: all data sources failed; using random tokens", flush=True)
        samples = [torch.randint(0, tokenizer.vocab_size, (seq_len,))
                   for _ in range(num_samples)]
    elif len(samples) < num_samples:
        real = len(samples)
        print(f"  WARNING: loaded only {real} real sequences; repeating to {num_samples}",
              flush=True)
        while len(samples) < num_samples:
            samples.append(samples[len(samples) % real].clone())

    return torch.stack(samples[:num_samples]).to(device)


def prepare_text_data(
    output: str | Path,
    num_samples: int,
    data_sources: str | Sequence[str] | None = None,
    data_preset: str = "public-mixed",
    source_timeout: int = 90,
    max_chars: int = 20000,
) -> dict[str, int]:
    """Export public/local text sources into a local JSONL file.

    This is intended for offline GPU environments: run it once on a machine
    with network access, move the JSONL file, then pass it via
    --calibration-data or --distill-data.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")

    sources = parse_data_sources(data_sources, preset=data_preset)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("  Data export pipeline: " + ", ".join(s.label for s in sources), flush=True)
    rows: list[dict[str, str]] = []
    loaded_by_source: dict[str, int] = {source.raw: 0 for source in sources}
    per_source = max(1, math.ceil(num_samples / len(sources)))

    for source in sources:
        remaining = num_samples - len(rows)
        if remaining <= 0:
            break
        target = min(per_source, remaining)
        new_rows = _collect_text_rows(
            source,
            target=target,
            skip_texts=loaded_by_source[source.raw],
            timeout=source_timeout,
            max_chars=max_chars,
        )
        loaded_by_source[source.raw] += len(new_rows)
        rows.extend(new_rows)

    if len(rows) < num_samples:
        for source in sources:
            if loaded_by_source[source.raw] == 0:
                continue
            remaining = num_samples - len(rows)
            if remaining <= 0:
                break
            new_rows = _collect_text_rows(
                source,
                target=remaining,
                skip_texts=loaded_by_source[source.raw],
                timeout=source_timeout,
                max_chars=max_chars,
            )
            loaded_by_source[source.raw] += len(new_rows)
            rows.extend(new_rows)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows[:num_samples]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Wrote {min(len(rows), num_samples)} rows to {output_path}", flush=True)
    return {source.label: loaded_by_source[source.raw] for source in sources}


def _parse_source(raw: str) -> DataSource:
    path = Path(raw).expanduser()
    suffix = path.suffix.lower()
    if path.exists() or suffix in {".jsonl", ".json", ".txt"}:
        return DataSource(raw=raw, path=str(path), local=True)

    parts = raw.split(":")
    if len(parts) == 1:
        return DataSource(raw=raw, path=raw)
    if len(parts) == 2:
        second = parts[1] or None
        if second in KNOWN_SPLITS:
            return DataSource(raw=raw, path=parts[0], split=second)
        return DataSource(raw=raw, path=parts[0], config=second)
    if len(parts) == 3:
        return DataSource(
            raw=raw,
            path=parts[0],
            config=parts[1] or None,
            split=parts[2] or "train",
        )
    raise ValueError(f"Invalid data source spec: {raw}")


def _load_from_source(
    source: DataSource,
    tokenizer,
    target: int,
    seq_len: int,
    skip_sequences: int = 0,
    timeout: int = 30,
) -> list[Tensor]:
    try:
        if source.local or timeout <= 0:
            examples = _iter_examples(source)
        else:
            max_examples = max((target + skip_sequences) * 4, 128)
            examples = _fetch_hf_texts_subprocess(source, max_examples, timeout)
        samples = _pack_examples(
            examples, tokenizer, target=target, seq_len=seq_len,
            skip_sequences=skip_sequences,
        )
        if samples:
            print(f"  Loaded data: {source.label} ({len(samples)} seqs)", flush=True)
        else:
            print(f"  Data source empty: {source.label}", flush=True)
        return samples
    except Exception as exc:
        print(f"  Data source failed: {source.label}: {exc}", flush=True)
        return []


def _collect_text_rows(
    source: DataSource,
    target: int,
    skip_texts: int = 0,
    timeout: int = 90,
    max_chars: int = 20000,
) -> list[dict[str, str]]:
    try:
        if source.local or timeout <= 0:
            texts = _collect_texts(_iter_examples(source), target, skip_texts, max_chars)
        else:
            texts = _fetch_hf_texts_subprocess(
                source,
                max_examples=target + skip_texts,
                timeout=timeout,
                max_chars=max_chars,
            )[skip_texts:skip_texts + target]
        rows = [{"text": text, "source": source.label} for text in texts]
        if rows:
            print(f"  Exported data: {source.label} ({len(rows)} rows)", flush=True)
        else:
            print(f"  Data source empty: {source.label}", flush=True)
        return rows
    except Exception as exc:
        print(f"  Data source failed: {source.label}: {exc}", flush=True)
        return []


def _fetch_hf_texts_subprocess(
    source: DataSource,
    max_examples: int,
    timeout: int,
    max_chars: int = 20000,
) -> list[str]:
    payload = json.dumps({
        "path": source.path,
        "config": source.config,
        "split": source.split,
        "max_examples": max_examples,
        "max_chars": max_chars,
    })
    cmd = [sys.executable, "-m", "draft_adapter.data", "--fetch-source-json", payload]
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(message[-500:] if message else f"exit code {result.returncode}")
    return json.loads(result.stdout)


def _iter_examples(source: DataSource) -> Iterator[Any]:
    if source.local:
        yield from _iter_local_examples(Path(source.path))
        return

    if source.config:
        ds = load_dataset(source.path, source.config, split=source.split, streaming=True)
    else:
        ds = load_dataset(source.path, split=source.split, streaming=True)
    yield from ds


def _fetch_hf_texts(source: DataSource, max_examples: int,
                    max_chars: int = 20000) -> list[str]:
    return _collect_texts(_iter_examples(source), max_examples, max_chars=max_chars)


def _collect_texts(
    examples: Iterable[Any],
    target: int,
    skip_texts: int = 0,
    max_chars: int = 20000,
) -> list[str]:
    texts: list[str] = []
    kept = 0
    for example in examples:
        text = _extract_text(example)
        if len(text.strip()) < 20:
            continue
        if kept < skip_texts:
            kept += 1
            continue
        texts.append(text[:max_chars])
        kept += 1
        if len(texts) >= target:
            break
    return texts


def _iter_local_examples(path: Path) -> Iterator[Any]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        yield from _iter_json_object(obj)
        return

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield {"text": line}


def _iter_json_object(obj: Any) -> Iterator[Any]:
    if isinstance(obj, list):
        yield from obj
        return
    if isinstance(obj, dict):
        for key in ("data", "train", "examples"):
            value = obj.get(key)
            if isinstance(value, list):
                yield from value
                return
        yield obj
        return
    yield {"text": str(obj)}


def _pack_examples(
    examples: Iterable[Any],
    tokenizer,
    target: int,
    seq_len: int,
    skip_sequences: int = 0,
) -> list[Tensor]:
    samples: list[Tensor] = []
    buffer: list[int] = []
    produced = 0
    eos_id = tokenizer.eos_token_id
    max_examples = max(target * 128, skip_sequences * 128, 1024)

    for seen, example in enumerate(examples):
        if seen >= max_examples and produced >= skip_sequences:
            break

        text = _extract_text(example)
        if len(text.strip()) < 20:
            continue

        ids = _tokenize(tokenizer, text, max_length=seq_len)
        if not ids:
            continue
        buffer.extend(ids)
        if eos_id is not None:
            buffer.append(int(eos_id))

        while len(buffer) >= seq_len:
            seq = torch.tensor(buffer[:seq_len], dtype=torch.long)
            del buffer[:seq_len]
            if produced < skip_sequences:
                produced += 1
                continue
            samples.append(seq)
            produced += 1
            if len(samples) >= target:
                return samples

    min_partial = min(seq_len, max(8, seq_len // 4))
    if len(samples) < target and len(buffer) >= min_partial:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        padded = buffer[:seq_len] + [int(pad_id)] * max(0, seq_len - len(buffer))
        seq = torch.tensor(padded[:seq_len], dtype=torch.long)
        if produced >= skip_sequences:
            samples.append(seq)
    return samples


def _tokenize(tokenizer, text: str, max_length: int) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(i) for i in input_ids]


def _extract_text(example: Any) -> str:
    if example is None:
        return ""
    if isinstance(example, str):
        return example
    if not isinstance(example, dict):
        return str(example)

    for field in ("text", "content"):
        text = _value_to_text(example.get(field))
        if text:
            return text

    for field in ("messages", "conversations"):
        text = _value_to_text(example.get(field))
        if text:
            return text

    parts: list[str] = []
    for field in (
        "system", "instruction", "prompt", "question", "input", "context",
        "answer", "response", "output", "chosen",
    ):
        text = _value_to_text(example.get(field))
        if text:
            parts.append(text)
    return "\n".join(parts)


def _value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for key in ("content", "value", "text", "prompt", "response", "output"):
            text = _value_to_text(value.get(key))
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(value, list):
        parts = [_value_to_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return str(value).strip()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser("draft-adapter-data")
    parser.add_argument("--fetch-source-json", default=None,
                        help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser(
        "prepare",
        help="Export public/local text data into local JSONL for offline use",
    )
    prepare.add_argument("--output", required=True,
                         help="Output JSONL path")
    prepare.add_argument("--num-samples", type=int, default=512,
                         help="Number of text rows to export")
    prepare.add_argument("--data-preset", default="public-zh-fast",
                         choices=sorted(DATA_PRESETS),
                         help="Built-in public data mix")
    prepare.add_argument("--data-sources", default=None,
                         help="Comma-separated local paths or HF specs "
                              "(dataset[:config[:split]])")
    prepare.add_argument("--source-timeout", type=int, default=90,
                         help="Seconds before skipping a slow HF source")
    prepare.add_argument("--max-chars", type=int, default=20000,
                         help="Maximum characters per exported row")

    args = parser.parse_args()

    if args.fetch_source_json is None and args.command is None:
        parser.print_help()
        raise SystemExit(2)

    if args.fetch_source_json is None and args.command == "prepare":
        counts = prepare_text_data(
            output=args.output,
            num_samples=args.num_samples,
            data_sources=args.data_sources,
            data_preset=args.data_preset,
            source_timeout=args.source_timeout,
            max_chars=args.max_chars,
        )
        print("  Source counts: " + json.dumps(counts, ensure_ascii=False), flush=True)
        return

    payload = json.loads(args.fetch_source_json)
    source = DataSource(
        raw=payload["path"],
        path=payload["path"],
        config=payload.get("config"),
        split=payload.get("split") or "train",
    )
    texts = _fetch_hf_texts(
        source,
        int(payload["max_examples"]),
        max_chars=int(payload.get("max_chars", 20000)),
    )
    print(json.dumps(texts, ensure_ascii=False))


if __name__ == "__main__":
    main()
