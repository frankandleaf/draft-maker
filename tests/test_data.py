"""Tests for public/local data loading."""

import json

import torch

from draft_adapter import data


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 256

    def __call__(self, text, add_special_tokens=False, truncation=True, max_length=None):
        ids = [(ord(ch) % 200) + 2 for ch in text if not ch.isspace()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_parse_hf_and_local_sources(tmp_path):
    local_path = tmp_path / "calib.jsonl"
    sources = data.parse_data_sources(
        f"allenai/c4:zh:train,Salesforce/wikitext:wikitext-2-raw-v1:train,{local_path}"
    )

    assert sources[0].path == "allenai/c4"
    assert sources[0].config == "zh"
    assert sources[0].split == "train"
    assert sources[1].path == "Salesforce/wikitext"
    assert sources[1].config == "wikitext-2-raw-v1"
    assert sources[2].local is True


def test_extracts_instruction_and_messages():
    instruction = data._extract_text({
        "instruction": "Explain SVD compression for language models.",
        "input": "Use concise wording.",
        "output": "SVD approximates large matrices with lower rank factors.",
    })
    messages = data._extract_text({
        "messages": [
            {"role": "user", "content": "Summarize the paragraph in Chinese."},
            {"role": "assistant", "content": "Here is a concise summary."},
        ]
    })

    assert "Explain SVD" in instruction
    assert "lower rank" in instruction
    assert "Summarize" in messages
    assert "concise summary" in messages


def test_loads_local_jsonl_as_packed_sequences(tmp_path):
    path = tmp_path / "calib.jsonl"
    rows = [
        {"text": "This is a public calibration sample with enough text."},
        {"instruction": "Write a Python function.", "output": "Return the result."},
        {"messages": [{"content": "Chinese assistant calibration text goes here."}]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    ids = data.load_calibration_data(
        FakeTokenizer(), num_samples=3, seq_len=8, device="cpu",
        data_sources=str(path),
    )

    assert ids.shape == (3, 8)
    assert ids.device.type == "cpu"
    assert torch.any(ids != 0)


def test_prepare_text_data_exports_local_jsonl(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "prepared" / "calib.jsonl"
    rows = [
        {"text": "First local calibration row with enough text to keep."},
        {"instruction": "Explain local offline preparation.",
         "output": "Write JSONL rows and read them later."},
        {"messages": [{"content": "Third local row stored as messages."}]},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    counts = data.prepare_text_data(
        output=output,
        num_samples=2,
        data_sources=str(source),
        source_timeout=0,
    )

    exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert sum(counts.values()) == 2
    assert len(exported) == 2
    assert exported[0]["source"] == str(source)
    assert "text" in exported[0]

    ids = data.load_calibration_data(
        FakeTokenizer(), num_samples=2, seq_len=8, device="cpu",
        data_sources=str(output),
    )
    assert ids.shape == (2, 8)


def test_loads_hf_spec_with_streaming(monkeypatch):
    calls = []

    def fake_load_dataset(path, *args, split=None, streaming=False):
        calls.append((path, args, split, streaming))
        return [
            {"text": "Streaming dataset sample with enough characters for packing."},
            {"text": "Another streaming dataset sample with enough characters."},
        ]

    monkeypatch.setattr(data, "load_dataset", fake_load_dataset)

    ids = data.load_calibration_data(
        FakeTokenizer(), num_samples=2, seq_len=8, device="cpu",
        data_sources="fake/corpus:cfg:train", source_timeout=0,
    )

    assert ids.shape == (2, 8)
    assert calls == [("fake/corpus", ("cfg",), "train", True)]
