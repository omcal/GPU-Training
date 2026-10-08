# GPU-Training

[![CI](https://github.com/omcal/GPU-Training/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/omcal/GPU-Training/actions/workflows/ci.yml)

A hands-on GPU programming notebook: CUDA C++ on NVIDIA hardware and Metal kernels
through MLX on Apple Silicon. Each lab combines kernel implementations, CPU reference
checks, and benchmarks to connect parallel algorithms with measured GPU behavior.

The curriculum follows *Programming Massively Parallel Processors* (PMPP).
The [PMPP + AI Systems Performance Engineering study plan](docs/08-cuda-inference-study-plan.md)
continues from execution and memory models to reduction, inference kernels, and runtime
experiments. The plan and study notes are in Turkish; the core lab commentary is in English.

**Start here:** [CUDA labs](cuda/README.md) · [Metal labs](labs/README.md)
· [Independent exercises](study/README.md) · [Documentation index](docs/README.md)

## What's implemented

| Topic | CUDA C++ | Metal / MLX | Focus |
|---|---|---|---|
| Vector add | [lab01](cuda/lab01_vector_add.cu) | [lab01](labs/lab01_vector_add.py) | Launch geometry, bounds, grid-stride loops |
| Coalescing / transpose | [lab02](cuda/lab02_coalescing.cu) | [lab02](labs/lab02_coalescing.py) | Memory access, shared-memory tiling |
| Reduction | [lab03](cuda/lab03_reduction.cu) | [lab03](labs/lab03_reduction.py) | Block cooperation, partial sums |
| Matrix multiplication | [lab04](cuda/lab04_matmul.cu) | [lab04](labs/lab04_matmul.py) | Data reuse, tiling, arithmetic intensity |
| Convolution / stencil | [lab05](cuda/lab05_convolution.cu) | [lab05](labs/lab05_convolution.py) | Halo regions, boundaries |
| Prefix sum / scan | [lab06](cuda/lab06_scan.cu) | [lab06](labs/lab06_scan.py) | Local scans, block totals, offsets |

[pmpp/](pmpp/README.md) also pairs vector-add and tiled-matmul examples in CUDA and
Metal. The two backends illustrate the same concepts; their optimization variants
and supported input shapes differ. Each lab documents its scope.

[study/](study/README.md) contains 12 independent exercise briefs, a 25-day schedule,
and daily/weekly note templates. Histogram, merge, softmax, RMSNorm, streams, and
CUDA Graphs are planned exercises; their solutions have not been implemented here.

## Quick start

```bash
git clone https://github.com/omcal/GPU-Training.git
cd GPU-Training
```

### NVIDIA / CUDA

The tested setup uses Ubuntu, CUDA Toolkit 12.9, and a GTX 1050 Ti Max-Q (`sm_61`).
Python and MLX are unnecessary for the CUDA labs.

```bash
make -C cuda test
# One lab, correctness checks only:
./build/cuda/lab01_vector_add --test
```

The Makefile defaults to `/usr/local/cuda-12.9`. On a different NVIDIA setup,
set `CUDA_HOME` to the toolkit directory and `ARCH` to the GPU target; use `make -B`
after changing compiler settings. See the [CUDA guide](cuda/README.md) for benchmarks,
sanitizer targets, and optional SSH synchronization.

### Apple Silicon / Metal

The Metal route uses Python 3.12, MLX, and the dependencies in
[pyproject.toml](pyproject.toml). With `uv` installed:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python labs/lab01_vector_add.py
.venv/bin/python -m pytest -q
```

CUDA kernels run on NVIDIA hardware; the Apple Silicon route runs Metal kernels.
A useful launch distinction: CUDA grid dimensions count blocks, while this MLX Metal
harness counts dispatched threads. [The translation guide](docs/01-cuda-to-metal.md)
explains the mapping and `gpuk.launch.cuda()` helper.

## Learning workflow

1. Read the matching chapter and state the problem, decomposition, and GPU constraints.
2. Predict the result, run the CPU reference checks, then measure the kernel.
3. Change one parameter and compare under the same timing protocol.
4. Rebuild a kernel from a blank file and explain the result in your own words.

For the 8 October–1 November 2026 CUDA program, start with the
[study guide](study/README.md) and record evidence in the [progress tracker](study/progress.md).
The [Metal learning route](docs/05-learning-route.md) remains available as a separate path.

## Validation and measurements

[Validation on 8 October 2026](docs/10-validation-2026-10-08.md): **40 Metal tests passed**;
**78 CUDA CPU-reference comparisons and two PMPP examples passed** on real hardware.
These results describe the included reference labs; the independent study exercises
remain to be completed.

Benchmark numbers are hardware- and workload-specific observations.
[Metal measurements](docs/02-your-hardware.md), [CUDA measurements](docs/07-cuda-validation.md),
and the [measurement guide](docs/03-how-to-measure.md) explain their context and limits.
Check correctness before recording timings; GPU kernel latency and end-to-end latency
are different metrics. Re-measure after changing hardware or experimental conditions.

## CI and releases

Pushes and pull requests validate Python syntax, local documentation links, benchmark
CSV structure, shell syntax, and workflow configuration. They also compile all CUDA labs
and PMPP examples with CUDA 12.9 for `sm_61`, uploading the compiled programs as artifacts.
GPU execution tests run separately on real hardware.

Version tags such as `v0.1.0` run the same checks, then publish source `.tar.gz` / `.zip`
archives and SHA-256 checksums to GitHub Releases. Details:
[CI and release guide](docs/09-ci-and-releases.md).

## Repository layout

```text
cuda/      CUDA C++ labs, Makefile, and shared helpers
labs/      Metal / MLX labs
gpuk/      Metal device, launch, timing, and correctness helpers
pmpp/      Paired CUDA and Metal reference examples
study/     Independent exercises, progress, notes, and benchmark CSV
docs/      Concepts, learning routes, hardware records, and validation
bench/     Metal peak-performance probes
tests/     Metal correctness suite
tools/     Repository checks, device probe, and optional SSH synchronization
.github/   CI, source-release workflow, and dependency updates
```

[Contributing](CONTRIBUTING.md) explains how to validate changes and report measurements.
