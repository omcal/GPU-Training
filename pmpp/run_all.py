#!/usr/bin/env python
"""Run the Metal kernels in ``pmpp/`` and check them.

The point of this script is that the ``.metal`` files are not decoration: it extracts
the marked kernel body from each file and actually compiles and runs it on the GPU. If a
``.metal`` file drifts out of sync with reality, this fails.

    python pmpp/run_all.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gpuk import assert_close, bandwidth_gbs, cuda, header, report  # noqa: E402
from gpuk.bench import best_ms  # noqa: E402
from gpuk.peaks import peak_flops_fp32  # noqa: E402

PMPP = REPO_ROOT / "pmpp"
BODY_RE = re.compile(
    r"//\s*--- MLX body begin ---(.*?)//\s*--- MLX body end ---", re.DOTALL
)


def extract_body(metal_path: Path) -> str:
    """Pull the MLX-runnable body out of a complete .metal file."""
    text = metal_path.read_text()
    match = BODY_RE.search(text)
    if not match:
        raise SystemExit(f"{metal_path}: no '--- MLX body begin/end ---' markers found")
    return match.group(1)


def run_vector_add() -> None:
    # 4M elements: big enough that the measurement is not dominated by launch overhead.
    # A 1M-element run here is latency bound and reports a misleadingly low bandwidth.
    n = 1 << 22
    src = extract_body(PMPP / "01_vector_add" / "kernel.metal")
    kernel = mx.fast.metal_kernel(
        name="pmpp_vector_add", input_names=["a", "b"], output_names=["c"], source=src
    )
    a = mx.random.normal(shape=(n,), dtype=mx.float32)
    b = mx.random.normal(shape=(n,), dtype=mx.float32)
    mx.eval(a, b)
    launch = cuda((n + 255) // 256, 256).kwargs()

    def run():
        # The launch must happen INSIDE the timed function. Timing `mx.eval(out)` on an
        # array that was already evaluated measures nothing at all -- see
        # docs/03-how-to-measure.md.
        return kernel(
            inputs=[a, b], output_shapes=[a.shape], output_dtypes=[a.dtype], **launch
        )[0]

    assert_close(run(), np.array(a) + np.array(b), label="pmpp vector_add")

    t = best_ms(lambda: mx.eval(run()), rounds=3, warmup=5, iters=20)
    nbytes = 3 * n * 4
    print(
        f"           {n:,} elements in {t.median:.4f} ms  "
        f"({bandwidth_gbs(nbytes, t.median):.1f} GB/s, 2R:1W)"
    )


def run_tiled_matmul() -> None:
    m = k = n = 256
    src = extract_body(PMPP / "02_tiled_matmul" / "kernel.metal")
    kernel = mx.fast.metal_kernel(
        name="pmpp_matmul_tiled", input_names=["A", "B"], output_names=["C"], source=src
    )
    a = mx.random.normal(shape=(m, k), dtype=mx.float32)
    b = mx.random.normal(shape=(k, n), dtype=mx.float32)
    mx.eval(a, b)
    launch = cuda((m // 16) * (n // 16), 256).kwargs()

    def run():
        return kernel(
            inputs=[a, b], output_shapes=[(m, n)], output_dtypes=[a.dtype], **launch
        )[0]

    reference = np.array(a).astype(np.float64) @ np.array(b).astype(np.float64)
    assert_close(run(), reference, rtol=1e-4, atol=1e-2, label="pmpp tiled_matmul")

    flops = 2.0 * m * n * k
    t = best_ms(lambda: mx.eval(run()), rounds=3, warmup=5, iters=20)
    peak = peak_flops_fp32()
    print(
        f"           {m}x{n}x{k}: {t.median:.4f} ms  "
        f"{flops / t.median * 1e-6:.1f} GFLOP/s ({100 * flops / t.median * 1e-6 / peak:.0f}% of fp32 peak)"
    )


def main() -> None:
    report()
    header("pmpp/01_vector_add   CUDA cloud: vector_add<<<blocks, 256>>>")
    run_vector_add()
    header("pmpp/02_tiled_matmul  CUDA cloud: matmul_tiled<<<blocks, 256>>>")
    run_tiled_matmul()
    print("\nBoth Metal ports of the PMPP kernels compiled, ran, and matched the reference.")


if __name__ == "__main__":
    main()
