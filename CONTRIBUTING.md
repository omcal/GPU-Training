# Contributing

This repository is a GPU learning notebook. Useful changes include correctness fixes,
clearer kernel explanations, boundary tests, and reproducible performance experiments.

## Set up the relevant backend

Follow the [CUDA guide](cuda/README.md) for NVIDIA C++ labs or the
[Metal quick start](README.md#apple-silicon--metal) for Python/MLX labs.
Use a branch for your change and keep its purpose focused.

## Validate a change

Portable checks from the repository root:

```bash
python3 tools/check_repo.py
bash -n tools/sync_cuda.sh
```

For Metal kernel changes, run the correctness suite on Apple Silicon:

```bash
.venv/bin/python -m pytest -q
```

For CUDA changes, compile and execute on a compatible NVIDIA GPU:

```bash
make -C cuda test
```

Choose `CUDA_HOME` and `ARCH` for your environment as described in the CUDA guide.
If the installed sanitizer supports that GPU, use the relevant `sanitize`, `racecheck`,
or `synccheck` target for memory and synchronization changes.
GitHub CI compiles CUDA without executing the kernels; report hardware test results
separately. Documentation-only changes need the portable checks.

## Kernel and benchmark changes

- Keep a CPU reference and test small, awkward/nonmultiple, and larger inputs.
- State the supported shapes, boundary behavior, and floating-point tolerances.
- Preserve explicit CUDA error handling and synchronization requirements.
- Measure after correctness checks; use the same protocol for compared variants.
- Record GPU, toolkit/framework versions, input shape, launch configuration, warm-up,
  repetition count, and units. Include power/thermal context when available.
- Explain whether timing covers a kernel, a pipeline, transfers, or an end-to-end path.
- Distinguish observed speed differences from hypotheses about their cause.

Use the [measurement guide](docs/03-how-to-measure.md) and
[benchmark CSV](study/benchmarks/results/results.csv) as references.
Leave unknown metadata blank and identify estimated traffic as an estimate.
Do not enforce timing thresholds in correctness tests.

## Independent study work

The [study exercises](study/README.md) are prompts for independent implementations.
Write the decomposition and first attempt before consulting reference solutions.
Update [progress](study/progress.md) with concrete evidence; running an existing lab
is separate from completing a learning objective. Use daily and weekly notes to record
what you can explain, implement, and measure.

## Pull request description

Describe the problem and resulting behavior. Include the test command and result,
the hardware used for GPU execution, and any affected shape or tolerance limits.
For performance changes, attach a reproducible baseline comparison and explain the
measurement method. Keep toolkit upgrades or new dependencies explicit.
