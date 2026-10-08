"""Correctness checking.

A GPU kernel that produces the wrong answer very often produces a *plausible* wrong
answer, so "it ran and printed numbers" means nothing. Every lab checks against a
reference before it measures speed.
"""

from __future__ import annotations

import numpy as np


def to_numpy(x) -> np.ndarray:
    """MLX array -> numpy array, forcing evaluation first."""
    import mlx.core as mx

    mx.eval(x)
    return np.array(x)


def max_abs_err(got, expected) -> float:
    """Largest absolute difference, with NaNs treated as disagreements."""
    a, b = to_numpy(got).astype(np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: got {a.shape}, expected {b.shape}")
    diff = np.abs(a - b)
    if np.isnan(diff).any():
        return float("inf")
    return float(diff.max())


def assert_close(
    got,
    expected,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    label: str = "kernel",
    verbose: bool = True,
) -> float:
    """Raise ``AssertionError`` unless ``got`` matches ``expected``; return max error.

    Tolerance defaults suit float32 accumulate-and-compare work. Matmul over a long
    reduction dimension legitimately needs a looser atol because float32 addition is
    not associative — see ``labs/lab04_matmul.py`` for that discussion.
    """
    a = to_numpy(got).astype(np.float64)
    b = np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shape mismatch: got {a.shape}, expected {b.shape}")

    err = float(np.abs(a - b).max()) if a.size else 0.0
    ok = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
    if verbose:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label:<28} max abs err = {err:.3e}  (rtol={rtol:g}, atol={atol:g})")
    if not ok:
        bad = np.argwhere(~np.isclose(a, b, rtol=rtol, atol=atol))
        preview = bad[:5].tolist() if bad.size else []
        raise AssertionError(
            f"{label}: mismatch, max abs err {err:.3e} at {preview} "
            f"(rtol={rtol:g}, atol={atol:g})"
        )
    return err
