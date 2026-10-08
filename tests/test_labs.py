"""Correctness tests for every kernel in the repo.

These run the *same kernel functions the labs use*, at small sizes, so the suite finishes
in seconds. They are deliberately not benchmarks — a test that asserts on wall-clock time
is a flaky test. Speed claims live in the labs, where the numbers are printed with their
caveats and never asserted on.

    python -m pytest
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gpuk import assert_close  # noqa: E402


def load_lab(filename: str):
    """Import a lab script as a module without running its main()."""
    path = REPO_ROOT / "labs" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lab01():
    return load_lab("lab01_vector_add.py")


@pytest.fixture(scope="module")
def lab02():
    return load_lab("lab02_coalescing.py")


@pytest.fixture(scope="module")
def lab03():
    return load_lab("lab03_reduction.py")


@pytest.fixture(scope="module")
def lab04():
    return load_lab("lab04_matmul.py")


@pytest.fixture(scope="module")
def lab05():
    return load_lab("lab05_convolution.py")


@pytest.fixture(scope="module")
def lab06():
    return load_lab("lab06_scan.py")


# ---------------------------------------------------------------------------- lab 01
@pytest.mark.parametrize("n", [256, 1000, 3000, 65536])
def test_lab01_vector_add(lab01, n):
    a = mx.random.normal(shape=(n,), dtype=mx.float32)
    b = mx.random.normal(shape=(n,), dtype=mx.float32)
    expected = np.array(a) + np.array(b)
    assert_close(lab01.add_per_element(a, b), expected, label=f"per_element n={n}", verbose=False)
    assert_close(lab01.add_grid_stride(a, b), expected, label=f"grid_stride n={n}", verbose=False)
    # A grid with a single block must still cover the whole array (grid-stride loop).
    assert_close(lab01.add_grid_stride(a, b, blocks=1), expected, label="1 block", verbose=False)


# ---------------------------------------------------------------------------- lab 02
@pytest.mark.parametrize("shape", [(64, 64), (64, 128), (128, 64)])
def test_lab02_transpose(lab02, shape):
    a = mx.random.normal(shape=shape, dtype=mx.float32)
    expected = np.array(a).T
    assert_close(lab02.transpose_naive(a), expected, label=f"naive {shape}", verbose=False)
    assert_close(lab02.transpose_tiled(a), expected, label=f"tiled {shape}", verbose=False)
    assert_close(lab02.transpose_big_tile(a), expected, label=f"big tile {shape}", verbose=False)


@pytest.mark.parametrize("shape", [(64, 96), (96, 64)])
def test_lab02_transpose_non_multiple_of_64(lab02, shape):
    """Only the 32x32 tiled kernel handles these; the 64x64 one must refuse."""
    a = mx.random.normal(shape=shape, dtype=mx.float32)
    expected = np.array(a).T
    assert_close(lab02.transpose_naive(a), expected, label=f"naive {shape}", verbose=False)
    assert_close(lab02.transpose_tiled(a), expected, label=f"tiled {shape}", verbose=False)
    with pytest.raises(ValueError, match="multiples of 64"):
        lab02.transpose_big_tile(a)


@pytest.mark.parametrize("stride", [1, 2, 4, 8])
def test_lab02_spaced_read(lab02, stride):
    n = 1 << 14
    a = mx.arange(0, n, dtype=mx.float32)
    got = lab02.spaced_read(a, stride)
    # Each simdgroup sums 32 values taken `stride` apart, then writes one float.
    expected = np.array(a).reshape(-1, 32 * stride)[:, ::stride].sum(axis=1)
    assert_close(got, expected, rtol=1e-5, atol=1e-3, label=f"spaced_read {stride}", verbose=False)


def test_lab02_tiled_requires_multiples_of_32(lab02):
    with pytest.raises(ValueError, match="multiples of 32"):
        lab02.transpose_tiled(mx.zeros((30, 64), dtype=mx.float32))


# ---------------------------------------------------------------------------- lab 03
@pytest.mark.parametrize("n", [1 << 14, 1 << 16])
def test_lab03_reductions(lab03, n):
    a = mx.random.normal(shape=(n,), dtype=mx.float32)
    expected = np.array([float(np.array(a).astype(np.float64).sum())])
    for name, fn in (
        ("v1", lab03.reduce_v1),
        ("v2", lab03.reduce_v2),
        ("v3", lab03.reduce_v3),
        ("v4", lab03.reduce_v4),
    ):
        # Relative tolerance: float32 sums in a non-deterministic order via atomics.
        assert_close(fn(a), expected, rtol=1e-4, atol=1e-2, label=f"reduce {name} n={n}", verbose=False)


@pytest.mark.parametrize("elems", [1, 2, 4, 8])
def test_lab03_ilp_sweep(lab03, elems):
    n = 1 << 14
    a = mx.random.normal(shape=(n,), dtype=mx.float32)
    expected = np.array([float(np.array(a).astype(np.float64).sum())])
    assert_close(
        lab03.reduce_v4(a, elems), expected, rtol=1e-4, atol=1e-2,
        label=f"reduce_v4 elems={elems}", verbose=False,
    )


# ---------------------------------------------------------------------------- lab 04
@pytest.mark.parametrize("size", [32, 64])
def test_lab04_matmul(lab04, size):
    m = k = n = size
    a = mx.random.normal(shape=(m, k), dtype=mx.float32)
    b = mx.random.normal(shape=(k, n), dtype=mx.float32)
    expected = np.array(a).astype(np.float64) @ np.array(b).astype(np.float64)
    assert_close(lab04.matmul_naive(a, b), expected, rtol=1e-4, atol=1e-3, label="naive", verbose=False)
    assert_close(lab04.matmul_tiled(a, b), expected, rtol=1e-4, atol=1e-3, label="tiled", verbose=False)
    assert_close(
        lab04.matmul_register_blocked(a, b), expected, rtol=1e-4, atol=1e-3,
        label="register-blocked", verbose=False,
    )


def test_lab04_non_square(lab04):
    """68x52x36 exercises ragged tiles that a square test would miss."""
    m, k, n = 96, 64, 64
    a = mx.random.normal(shape=(m, k), dtype=mx.float32)
    b = mx.random.normal(shape=(k, n), dtype=mx.float32)
    expected = np.array(a).astype(np.float64) @ np.array(b).astype(np.float64)
    assert_close(lab04.matmul_tiled(a, b), expected, rtol=1e-4, atol=1e-3, label="tiled 96x64x64", verbose=False)
    assert_close(
        lab04.matmul_register_blocked(a, b), expected, rtol=1e-4, atol=1e-3,
        label="reg 96x64x64", verbose=False,
    )


# ---------------------------------------------------------------------------- lab 05
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("shape", [(64, 64), (40, 56), (33, 17), (1, 1)])
def test_lab05_convolution(lab05, k, shape):
    """Ragged shapes are the point: they are what exercises the halo bounds check."""
    img = np.random.default_rng(k).normal(size=shape).astype(np.float32)
    mx_img = mx.array(img)
    filt = lab05.pyramid_filter(k)
    mx_filt = mx.array(filt.reshape(-1))
    mx.eval(mx_img, mx_filt)
    expected = lab05.reference_conv(img, filt)
    assert_close(lab05.conv_naive(mx_img, mx_filt, k), expected, rtol=1e-4, atol=1e-4,
                 label=f"naive K={k} {shape}", verbose=False)
    assert_close(lab05.conv_naive_smem(mx_img, mx_filt, k), expected, rtol=1e-4, atol=1e-4,
                 label=f"naive+smem K={k} {shape}", verbose=False)
    assert_close(lab05.conv_tiled(mx_img, mx_filt, k), expected, rtol=1e-4, atol=1e-4,
                 label=f"tiled K={k} {shape}", verbose=False)


def test_lab05_filter_is_separable_and_asymmetric(lab05):
    """Asymmetry is a testing requirement, not an aesthetic choice."""
    for k in (1, 2, 4):
        f = lab05.ramp_filter(k)
        assert f.shape == (2 * k + 1, 2 * k + 1)
        assert abs(float(f.sum()) - 1.0) < 1e-6
        # Asymmetric: a transposed filter index would give a different answer.
        assert f[0, :].tolist() != f[:, 0].tolist()
        # Rank 1: exactly separable. The explicit tolerance is required because the
        # filter is stored as float32, which perturbs the zero singular value to ~1e-8.
        assert np.linalg.matrix_rank(f.astype(np.float64), tol=1e-6) == 1


def test_lab05_launch_covers_a_full_threadgroup(lab05):
    """A barrier inside a partially-dispatched threadgroup is undefined behaviour."""
    for h, w in ((1, 1), (1, 100), (100, 1), (5, 5), (2048, 2048)):
        launch = lab05.launch_for(h, w)
        assert launch["grid"][0] >= launch["threadgroup"][0]
        assert launch["grid"][1] >= launch["threadgroup"][1]


def test_lab05_copy_floor_runs(lab05):
    img = mx.random.normal(shape=(37, 41), dtype=mx.float32)
    mx.eval(img)
    assert_close(lab05.copy_floor(img), np.array(img), label="copy floor", verbose=False)


# ---------------------------------------------------------------------------- lab 06
def test_lab06_single_block_scans(lab06):
    n = lab06.BLOCK
    x = mx.random.normal(shape=(n,), dtype=mx.float32)
    mx.eval(x)
    x_np = np.array(x)
    assert_close(
        lab06.scan_hillis_steele(x), np.cumsum(x_np.astype(np.float64)),
        rtol=1e-5, atol=1e-4, label="hillis-steele", verbose=False,
    )
    scanned, total = lab06.scan_blelloch(x)
    assert_close(scanned, lab06.exclusive_reference(x_np), rtol=1e-5, atol=1e-4,
                 label="blelloch", verbose=False)
    expected_total = float(np.cumsum(x_np.astype(np.float64))[-1])
    assert abs(float(np.array(total)[0]) - expected_total) < 1e-3 * max(1.0, abs(expected_total))


@pytest.mark.parametrize("nblocks", [1, 4, 16])
def test_lab06_hierarchical_scan(lab06, nblocks):
    n = lab06.BLOCK * nblocks
    x = mx.random.normal(shape=(n,), dtype=mx.float32)
    mx.eval(x)
    expected = lab06.exclusive_reference(np.array(x))
    assert_close(lab06.scan_hierarchical(x), expected, rtol=1e-3, atol=1e-3,
                 label=f"hierarchical nblocks={nblocks}", verbose=False)


def test_lab06_rejects_unsupported_sizes(lab06):
    with pytest.raises(ValueError, match="power of two"):
        lab06.scan_hierarchical(mx.zeros((lab06.BLOCK * 3,), dtype=mx.float32))


# ---------------------------------------------------------------------------- pmpp/
def test_pmpp_kernels_run():
    """The .metal reference files must actually compile and be correct."""
    sys.path.insert(0, str(REPO_ROOT / "pmpp"))
    import run_all  # type: ignore[import-not-found]

    run_all.run_vector_add()
    run_all.run_tiled_matmul()
