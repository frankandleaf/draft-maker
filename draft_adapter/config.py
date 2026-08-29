"""Configuration for the small utilities retained in Draft-Adapter."""

from dataclasses import dataclass


@dataclass
class DistillConfig:
    """Experimental settings for the retained distillation scripts.

    This dataclass is kept for configuration compatibility with old research
    notebooks.  It is not used by the benchmark or CLI and is intentionally
    not a model-generation pipeline.
    """

    steps: int = 4000
    batch_size: int = 4
    max_seq_len: int = 512
    learning_rate: float = 5e-5
    top_k: int = 10
    kl_temperature: float = 1.0
    kl_mode: str = "reverse"
    hard_label_weight: float = 1.0
    num_train_prompts: int = 128
    generate_len: int = 32

    def __post_init__(self) -> None:
        if self.kl_mode not in ("reverse", "forward", "tvd"):
            raise ValueError(
                "kl_mode must be 'reverse', 'forward', or 'tvd', "
                f"got {self.kl_mode}"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.hard_label_weight < 0:
            raise ValueError(
                "hard_label_weight must be >= 0, "
                f"got {self.hard_label_weight}"
            )
