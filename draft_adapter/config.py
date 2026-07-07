"""Configuration dataclasses for the Draft-Adapter pipeline."""

from dataclasses import dataclass, field


@dataclass
class WidthConfig:
    """Width compression factors (0.0–1.0 multipliers).

    head_dim is FROZEN — changing it breaks RoPE position encoding.
    Width reduction is absorbed entirely by num_heads.

    Attributes:
        embed_size_factor: Scales hidden_size / embed_dim and intermediate_size.
        calibration_samples: Number of calibration sequences for PCA.
        calibration_seq_len: Max tokens per calibration sequence.
    """

    embed_size_factor: float = 0.5
    calibration_samples: int = 16
    calibration_seq_len: int = 512

    def __post_init__(self):
        if not 0 < self.embed_size_factor <= 1:
            raise ValueError(
                f"embed_size_factor must be in (0, 1], got {self.embed_size_factor}"
            )


@dataclass
class DepthConfig:
    """Depth pruning parameters.

    Attributes:
        layer_factor: Fraction of layers to keep. Range (0, 1].
        protect_first: Number of initial layers always kept.
        protect_last: Number of final layers always kept.
    """

    layer_factor: float = 0.75  # es*ls = 0.5*0.75 ≈ 0.375
    protect_first: int = 1
    protect_last: int = 1

    def __post_init__(self):
        if not 0 < self.layer_factor <= 1:
            raise ValueError(f"layer_factor must be in (0, 1], got {self.layer_factor}")
        if self.protect_first < 0 or self.protect_last < 0:
            raise ValueError("protect_first and protect_last must be >= 0")


@dataclass
class DistillConfig:
    """Distillation hyperparameters.

    Attributes:
        steps: Number of training steps.
        batch_size: Batch size per step.
        max_seq_len: Maximum sequence length for on-policy generation.
        learning_rate: AdamW learning rate.
        top_k: Top-K for sparse KL divergence.
        kl_temperature: Temperature for softening logits.
        kl_mode: "reverse", "forward", or "tvd".
        num_train_prompts: Number of training prompts.
        generate_len: Number of tokens student generates per step.
    """

    steps: int = 4000
    batch_size: int = 4
    max_seq_len: int = 512
    learning_rate: float = 5e-5
    top_k: int = 10
    kl_temperature: float = 1.0
    kl_mode: str = "reverse"
    num_train_prompts: int = 128
    generate_len: int = 32

    def __post_init__(self):
        if self.kl_mode not in ("reverse", "forward", "tvd"):
            raise ValueError(f"kl_mode must be 'reverse', 'forward', or 'tvd', got {self.kl_mode}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration.

    Attributes:
        model: HuggingFace model ID or local path.
        tokenizer: HuggingFace tokenizer ID (defaults to model).
        output: Output directory for the draft model.
        device: Torch device string.
        dtype: Model dtype string.
        seed: Random seed for reproducibility.
        width: Width compression config.
        depth: Depth pruning config.
        distill: Distillation config (None = use defaults if --distill).
        skip_distill: Skip the distillation step.
        skip_benchmark: Skip the vLLM benchmark step.
    """

    model: str
    tokenizer: str | None = None
    output: str = "./draft_model"
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42
    width: WidthConfig = field(default_factory=WidthConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    distill: DistillConfig | None = None
    skip_distill: bool = False
    skip_benchmark: bool = False
    debug: bool = False
    method: str = "slicegpt"

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model
