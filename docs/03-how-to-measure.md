# 03 · How to measure a kernel without fooling yourself

Most "GPU optimisation" stories are measurement errors. This page is the checklist that
keeps you honest, plus the workflow for deciding what to optimise.

---

## The four traps

### 1. Not synchronising

GPU work is asynchronous and MLX is lazy. This measures nothing:

```python
t0 = time.perf_counter()
out = kernel(...)          # only records a node in the graph
t1 = time.perf_counter()   # ~0.01 ms — the GPU has not started
```

You must force evaluation and wait:

```python
mx.eval(out)               # or mx.synchronize()
```

Use `gpuk.bench.time_ms`, which synchronises inside the timed loop.

### 2. Not warming up

The first dispatch of a Metal kernel builds and compiles a shader library. This can take
milliseconds — orders of magnitude more than the kernel itself. Averaged into 20
iterations it can understate bandwidth by 30%+.

Every benchmark in this repo warms up at least 5 iterations. When I first wrote
`gpuk/peaks.py` with `warmup=3` it reported **49 GB/s** for a kernel that actually runs
at ~92 GB/s. That is the size of the error this trap produces.

### 3. Measuring a working set that is too small

Small kernels are launch-bound, not bandwidth-bound. Measured on this machine with a
read-only probe that is verified against the array's true sum:

| Working set | Effective read bandwidth |
|---|---|
| 4 MiB | ~27 GB/s |
| 64 MiB | ~59 GB/s |
| 256 MiB | ~96 GB/s |

A 4 MiB streaming kernel gets about **a quarter** of the asymptotic bandwidth. If you
measure your kernel at 64 MiB and compare it against a 256 MiB probe, you will conclude
your kernel is bad when it is merely small. Match the working set.

There is a second-order trap in the same table: the numbers only mean something if the
kernel actually read all the data. A probe with a divergent `simd_sum` once reported
167 GB/s on this 120 GB/s machine — see the bug write-up in
[`02-your-hardware.md`](02-your-hardware.md). **A bandwidth number with no correctness
check next to it is not a measurement.**

### 4. Thermal throttling and background load

This is a fanless MacBook Air. Under sustained GPU load it throttles, and two consecutive
runs of the identical kernel have differed by 20% during development. A browser
compositing a window steals both bandwidth and GPU time.

Mitigations:

* `gpuk.bench.best_ms(fn, rounds=3)` — repeat the whole measurement, keep the fastest
  round.
* Run benchmarks on AC power, with nothing else heavy running.
* Treat differences under ~10% as ties. Do not build a conclusion on one run.

---

## Latency vs throughput: two different numbers

These answer different questions and must not be compared:

| | `gpuk.bench.time_ms` | `gpuk.bench.throughput_ms` |
|---|---|---|
| synchronises | every iteration | once at the end |
| measures | cost of one call, *including launch overhead* | steady-state GPU cost per call |
| good for | "how fast is one inference of this layer" | "how fast is this kernel in a loop" |
| kernel must be | any size | large enough that the GPU is the bottleneck |

For kernels above ~0.5 ms they agree. Below that they diverge, and the gap *is* the
launch overhead. Always say which one a number came from.

---

## The workflow: measure, classify, then optimise

**Step 1 — establish correctness first.** A wrong kernel's speed is meaningless. Every
lab here calls `gpuk.assert_close` against a reference before benchmarking. Use a
`float64` reference for anything with a long reduction (see below).

**Step 2 — measure the ceilings on the machine you are on.** Never compare against a
spec sheet.

```python
from gpuk.peaks import peaks
print(peaks())   # {'bandwidth_1r1w_gbs': ..., 'bandwidth_2r1w_gbs': ..., 'flops_fp32': ...}
```

**Step 3 — classify the kernel with arithmetic intensity.**

```
intensity = FLOPs / bytes moved
crossover = peak_flops / peak_bandwidth      # ≈ 26 FLOP/byte here
```

* Below the crossover → **memory bound**. Only fewer bytes help. Fusion, better access
  patterns, smaller dtypes, tiling for reuse. No arithmetic change will show up.
* Above the crossover → **compute bound**. Only fewer/cheaper FLOPs help. Vectorisation,
  register blocking, `simdgroup_matrix`, lower precision.

Optimising the wrong side produces a flat line. This is the most common wasted week in
GPU work.

**Step 4 — compare against the best available implementation.** "Faster than my last
version" is not a result. `mx.add`, `mx.sum`, `mx.matmul` are the baselines in the labs,
and losing to them is normal and informative — labs 01-03 match them, lab 04 does not.

**Step 5 — sweep the one parameter that matters.** Grid size, tile size, elements per
thread. `labs/lab01` sweeps block count; `labs/lab03` sweeps elements per thread;
`labs/lab02` compares tile sizes. A sweep that comes out *flat* is a real result: it
means that knob is not your bottleneck.

---

## Float32 does not add associatively

A GPU reduction or dot product sums in a different order every run. Your kernel and a
reference will differ in the last few bits, and the difference grows with the length of
the reduction. This is not a bug.

```python
# right: relative comparison against a float64 reference
assert_close(kernel_out, np.float64_reference, rtol=1e-4, atol=1e-2)

# wrong: expecting bit equality
assert (kernel_out == reference).all()
```

`labs/lab04` uses `rtol=1e-4, atol=1e-2` for a K=1024 float32 matmul. Tightening that
tolerance is not rigour; it is a misunderstanding.

---

## Getting the generated Metal source

The fastest way to debug a kernel is to read what MLX actually compiled:

```python
out = kernel(inputs=[x], ..., grid=(...), threadgroup=(...), verbose=True)[0]
```

This prints the full MSL, including the signature MLX generated, the buffer indices, and
the template instantiation. If your kernel "does nothing", the reason is usually visible
here in about five seconds.

---

## Profiling: the next level

The labs here use wall-clock timing, which tells you *that* a kernel is slow but not
*why*. To see why, you need a profiler.

**Metal Debugger (needs full Xcode).** MLX can capture a trace:

```python
import mlx.core as mx
mx.metal.start_capture("trace.gputrace")
# ... run your kernel ...
mx.metal.stop_capture()
```

Open the resulting `.gputrace` in Xcode's Metal Debugger. You get per-kernel timings, the
shader's occupancy, threadgroup memory usage, and a per-line cost annotation of the
shader source. This is the Apple equivalent of Nsight Compute, and it is the tool that
answers "am I occupancy limited?" definitively.

**Note on this machine's setup:** only the Xcode *Command Line Tools* are installed, so
`xcrun metal` (the offline shader compiler) does not exist. That is fine for every lab in
this repo — MLX compiles Metal at runtime through the OS's `Metal.framework`, so no Xcode
is needed to write and run kernels. Install full Xcode only when you want the offline
compiler (`metal` / `metallib`) or the Metal Debugger.

**Command-line alternatives** if you do not want Xcode:

* `powermetrics --samplers gpu_power -i 1000` — GPU utilisation and power, system-wide.
* `sudo powermetrics` will also show GPU frequency residency, which reveals throttling.

---

## A pre-flight checklist

Before you believe any GPU number, including your own:

- [ ] Correctness checked against a reference, with a sane tolerance.
- [ ] `mx.eval`/`synchronize` inside the timing.
- [ ] Warm-up iterations ≥ 5.
- [ ] Working set comparable to the baseline's.
- [ ] On AC power, nothing else using the GPU.
- [ ] Repeated (best-of-N), and the spread reported.
- [ ] Arithmetic intensity computed, and the bottleneck named.
- [ ] Compared against the best available implementation, not just the previous version.
- [ ] The one parameter that matters swept, not guessed.

---

*Next: [`04-pmpp-roadmap.md`](04-pmpp-roadmap.md).*
