"""Verify calibration data loading logic — check if real text is loaded."""
import torch
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"


def check_source(name: str, loader_fn, tokenizer, num_samples=4, seq_len=32):
    """Test a single data source. Returns (ok, first_text)."""
    try:
        ds = loader_fn()
        samples = []
        for i, example in enumerate(ds):
            if len(samples) >= num_samples:
                break
            text = example.get("text", example.get("content", ""))
            if not text or len(text.strip()) < 20:
                continue
            tokens = tokenizer(text, truncation=True, max_length=seq_len,
                               return_tensors="pt")
            ids = tokens.input_ids[0]
            if ids.shape[0] < seq_len:
                ids = torch.nn.functional.pad(
                    ids, (0, seq_len - ids.shape[0]),
                    value=tokenizer.pad_token_id or 0)
            samples.append(ids[:seq_len])
        if samples:
            result = torch.stack(samples)
            decoded = tokenizer.decode(samples[0])
            return True, decoded, result.shape
        return False, "no valid samples", None
    except Exception as e:
        return False, str(e), None


if __name__ == "__main__":
    print(f"Loading tokenizer: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    from datasets import load_dataset

    sources = [
        ("allenai/c4-zh", lambda: load_dataset(
            "allenai/c4", "zh", split="train", streaming=True)),
        ("allenai/c4-en", lambda: load_dataset(
            "allenai/c4", "en", split="train", streaming=True)),
        ("wikitext", lambda: load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="train", streaming=True)),
    ]

    print(f"\n{'='*60}")
    print(f"Testing {len(sources)} data sources...")
    print(f"{'='*60}")

    any_ok = False
    for name, loader in sources:
        ok, msg, shape = check_source(name, loader, tok)
        status = "✓ OK" if ok else "✗ FAIL"
        print(f"\n{status}  {name}")
        if ok:
            any_ok = True
            print(f"       shape: {shape}")
            print(f"       text:  \"{msg[:120]}...\"")
        else:
            print(f"       error: {msg[:200]}")

    print(f"\n{'='*60}")
    if any_ok:
        print("PASS: at least one data source works → PCA will be meaningful")
    else:
        print("FAIL: all data sources failed → random tokens → PCA is noise → MODEL WILL BREAK")
        print("FIX: pip install datasets, or provide a local text file with --calibration-data")
    print(f"{'='*60}")
