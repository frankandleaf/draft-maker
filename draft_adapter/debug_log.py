"""Debug logging utilities — toggle with PipelineConfig.debug=True."""

import torch
from torch import Tensor


class DebugLogger:
    """Conditional debug printing. Enabled via PipelineConfig.debug."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._indent = 0

    def section(self, title: str) -> None:
        if not self.enabled:
            return
        print(f"\n{'='*60}")
        print(f"[DEBUG] {title}")
        print(f"{'='*60}")

    def info(self, msg: str) -> None:
        if not self.enabled:
            return
        pad = "  " * self._indent
        print(f"{pad}[DEBUG] {msg}")

    def before(self, op: str, **kwargs) -> None:
        """Log before an operation."""
        if not self.enabled:
            return
        pad = "  " * self._indent
        print(f"{pad}[DEBUG] ▶ {op}")
        for k, v in kwargs.items():
            if isinstance(v, Tensor):
                print(f"{pad}         {k}: shape={list(v.shape)}, "
                      f"dtype={v.dtype}, device={v.device}, "
                      f"norm={v.float().norm():.2f}")
            elif isinstance(v, (list, tuple)) and len(v) < 20:
                print(f"{pad}         {k}: {v}")
            elif isinstance(v, (int, float, str)):
                print(f"{pad}         {k}: {v}")
            else:
                print(f"{pad}         {k}: {type(v).__name__}")

    def after(self, result, **kwargs) -> None:
        """Log after an operation."""
        if not self.enabled:
            return
        pad = "  " * self._indent
        if isinstance(result, Tensor):
            print(f"{pad}         → shape={list(result.shape)}, "
                  f"norm={result.float().norm():.2f}")
        elif isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, Tensor):
                    print(f"{pad}         → {k}: shape={list(v.shape)}")
                else:
                    print(f"{pad}         → {k}: {v}")
        for k, v in kwargs.items():
            if isinstance(v, Tensor):
                print(f"{pad}         → {k}: shape={list(v.shape)}")
            else:
                print(f"{pad}         → {k}: {v}")

    def weight_diff(self, name: str, before: Tensor, after: Tensor) -> None:
        """Log weight changes: shape, norm, min, max before/after."""
        if not self.enabled:
            return
        pad = "  " * self._indent
        print(f"{pad}[DEBUG] Δ {name}:")
        print(f"{pad}         before: shape={list(before.shape)}, "
              f"norm={before.float().norm():.2f}, "
              f"min={before.float().min():.4f}, max={before.float().max():.4f}")
        print(f"{pad}         after:  shape={list(after.shape)}, "
              f"norm={after.float().norm():.2f}, "
              f"min={after.float().min():.4f}, max={after.float().max():.4f}")

    def indent(self) -> None:
        self._indent += 1

    def dedent(self) -> None:
        self._indent = max(0, self._indent - 1)


# Singleton
_logger: DebugLogger = DebugLogger(False)


def get_logger() -> DebugLogger:
    return _logger


def enable_debug() -> None:
    _logger.enabled = True


def disable_debug() -> None:
    _logger.enabled = False
