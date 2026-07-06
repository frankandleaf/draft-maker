"""Architecture inspection and target dimension computation."""

from dataclasses import dataclass

from transformers import AutoConfig

from .config import DepthConfig, WidthConfig

# Whitelist of supported GQA decoder architectures
SUPPORTED_ARCHITECTURES = {"llama", "mistral", "qwen2", "qwen3", "gemma2", "stablelm"}


class UnsupportedArchitectureError(ValueError):
    """Raised when the model architecture is not supported."""

    def __init__(self, model_type: str):
        super().__init__(
            f"Unsupported model type '{model_type}'. "
            f"Only GQA decoder architectures are supported: {sorted(SUPPORTED_ARCHITECTURES)}"
        )


@dataclass
class ModelArchitecture:
    """Extracted architecture parameters from a HF model config.

    Attributes are the ORIGINAL model parameters unless prefixed with 'target_'.
    """

    model_type: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    tie_word_embeddings: bool

    # Computed target dimensions (populated by compute_targets)
    target_embed_dim: int | None = None
    target_head_dim: int | None = None
    target_num_heads: int | None = None
    target_num_kv_heads: int | None = None
    target_intermediate_size: int | None = None
    target_num_layers: int | None = None

    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per key-value head."""
        return self.num_attention_heads // self.num_kv_heads


def inspect_model(model_name_or_path: str) -> ModelArchitecture:
    """Load a HF model config and extract architecture parameters.

    Args:
        model_name_or_path: HF model ID or local path.

    Returns:
        ModelArchitecture with original model parameters.

    Raises:
        UnsupportedArchitectureError: If the model type is not in the whitelist.
    """
    config = AutoConfig.from_pretrained(model_name_or_path)
    model_type = getattr(config, "model_type", "")

    # Normalize model type variants
    if model_type.startswith("qwen2") and model_type != "qwen3":
        model_type = "qwen2"  # qwen2, qwen2_5, etc. -> qwen2

    if model_type not in SUPPORTED_ARCHITECTURES:
        raise UnsupportedArchitectureError(model_type)

    # Extract architecture parameters with defaults per model family
    head_dim = _get_head_dim(config, model_type)
    intermediate_size = getattr(config, "intermediate_size", 0)

    return ModelArchitecture(
        model_type=model_type,
        num_layers=getattr(config, "num_hidden_layers", 0),
        num_attention_heads=getattr(config, "num_attention_heads", 0),
        num_kv_heads=getattr(config, "num_key_value_heads",
                              getattr(config, "num_attention_heads", 0)),
        head_dim=head_dim,
        hidden_size=getattr(config, "hidden_size", 0),
        intermediate_size=intermediate_size,
        vocab_size=getattr(config, "vocab_size", 0),
        max_position_embeddings=getattr(config, "max_position_embeddings", 0),
        rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
    )


def _get_head_dim(config, model_type: str) -> int:
    """Extract head_dim from config, with model-specific fallbacks."""
    # Qwen2/Qwen3/Llama configs expose head_dim directly (added in transformers 4.45+)
    if hasattr(config, "head_dim"):
        return config.head_dim
    # Compute from hidden_size / num_attention_heads
    num_heads = getattr(config, "num_attention_heads", 0)
    hidden_size = getattr(config, "hidden_size", 0)
    if num_heads > 0 and hidden_size > 0:
        return hidden_size // num_heads
    return 0


def compute_targets(arch: ModelArchitecture,
                    width: WidthConfig,
                    depth: DepthConfig) -> ModelArchitecture:
    """Compute target dimensions for the draft model.

    Applies width and depth scaling factors, then validates GQA constraints
    and adjusts dimensions to ensure divisibility.

    Key constraints enforced:
      1. target_num_heads * target_head_dim == target_embed_dim
      2. target_num_heads % target_num_kv_heads == 0 (GQA invariant)

    Args:
        arch: Original model architecture.
        width: Width compression config.
        depth: Depth pruning config.

    Returns:
        ModelArchitecture with target_* fields populated.
        head_dim is FROZEN — changing it breaks RoPE position encoding.
        Width reduction is absorbed entirely by num_heads reduction.
    """
    # Stage 1: head_dim is FROZEN (RoPE depends on it)
    target_head_dim = arch.head_dim

    # Stage 2: Scale embed_dim, then derive num_heads from it
    target_embed_dim = max(target_head_dim, int(arch.hidden_size * width.embed_size_factor))
    # Make embed_dim a multiple of head_dim
    target_embed_dim = (target_embed_dim // target_head_dim) * target_head_dim
    target_num_heads = target_embed_dim // target_head_dim

    # Stage 3: Scale intermediate_size and num_layers
    target_intermediate_size = max(8, int(arch.intermediate_size * width.embed_size_factor))
    target_num_layers = max(1, int(arch.num_layers * depth.layer_factor))

    # Stage 4: Enforce GQA invariant (num_heads % num_kv_heads == 0)
    kv_groups = arch.num_attention_heads // arch.num_kv_heads
    # Scale num_kv_heads proportionally to num_heads
    target_num_kv_heads = max(1, target_num_heads // kv_groups)

    # Ensure target_num_heads is a multiple of target_num_kv_heads
    if target_num_heads % target_num_kv_heads != 0:
        target_num_heads = (target_num_heads // target_num_kv_heads) * target_num_kv_heads
        target_embed_dim = target_num_heads * target_head_dim

    # Stage 5: Ensure at least protect_first + protect_last layers remain
    min_layers = depth.protect_first + depth.protect_last
    target_num_layers = max(target_num_layers, min_layers)

    # Stage 6: Validate
    assert target_num_heads * target_head_dim == target_embed_dim, \
        f"Attention dim mismatch: {target_num_heads} * {target_head_dim} != {target_embed_dim}"
    assert target_num_heads % target_num_kv_heads == 0, \
        f"GQA constraint violated: {target_num_heads} % {target_num_kv_heads} != 0"

    arch.target_embed_dim = target_embed_dim
    arch.target_head_dim = target_head_dim
    arch.target_num_heads = target_num_heads
    arch.target_num_kv_heads = target_num_kv_heads
    arch.target_intermediate_size = target_intermediate_size
    arch.target_num_layers = target_num_layers

    return arch
