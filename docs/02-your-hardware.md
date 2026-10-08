# 02 · Your hardware: what this M4 can actually do

Everything here was measured on the machine this repo was built on, and can be
re-measured by you at any time. Trust the commands over the numbers in this file: your
Mac will have a different thermal history, and laptop numbers move.

```bash
python -m gpuk              # GPU report: memory, limits, MLX version
python bench/m4_peaks.py    # re-measure the ceilings below
swift -module-cache-path .swiftcache tools/device_probe.swift   # raw Metal limits
```

---

## The machine

```
MacBook Air (Mac16,12), Apple M4, 10 CPU cores (4P + 6E), 16 GiB unified memory
macOS 26.6.2 · Metal 4 · Apple GPU family apple9 · MLX 0.32.2 · Python 3.12
```

## GPU limits, straight from `MTLDevice`

`tools/device_probe.swift` asks Metal itself rather than a spec sheet:

| Property | Value | Why you care |
|---|---|---|
| GPU cores | **8** | 8 × 128 = 1024 FP32 lanes |
| `maxThreadsPerThreadgroup` | **1024** | a 32×32 tile exactly saturates it |
| `maxThreadgroupMemoryLength` | **32768 B (32 KiB)** | shared with L1; the tiling budget |
| `maxBufferLength` | **8.88 GiB** | no single array may be larger |
| `recommendedMaxWorkingSetSize` | **11.84 GiB** | the GPU's real budget of the 16 GiB |
| `hasUnifiedMemory` | `true` | no host↔device copies exist |
| `supportsFamily(.apple9)` | `true` | M4-generation GPU features available |
| `supportsFamily(.metal4)` | `true` | Metal 4 features available |
| `argumentBuffersTier` | 1 | argument buffers, but not tier-2 features |

Note `maxThreadgroupMemoryLength` is the number that constrains almost every tiled
kernel you will write, and it is **not** in any Python API — hence the Swift probe.
It is also why `labs/lab02_coalescing.py` can show a 64×64 tile being 5.9x slower than a
32×32 one.

## Measured ceilings

Probed with large working sets and proper warm-up (see `docs/03-how-to-measure.md`).

| Probe | Measured | Notes |
|---|---|---|
| Streaming **1 read : 1 write** (`mx.maximum`, 512 MiB traffic) | **≈ 92-98 GB/s** | copy-shaped work |
| Streaming **2 reads : 1 write** (`mx.add`) | **≈ 94-97 GB/s** | the ceiling vector add chases |
| Coalesced **read-only**, 256 MiB working set, 16 loads/thread | **≈ 96 GB/s** | verified against the array's true sum |
| Coalesced **read-only**, 64 MiB working set | ≈ 59 GB/s | small working sets cannot saturate DRAM |
| **FP32 GEMM** (`mx.matmul`, 2048³) | **≈ 2300-2400 GFLOP/s** | uses `simdgroup_matrix` |
| FP32 GEMM (1024³) | ≈ 1800 GFLOP/s | smaller problems expose overhead |

Roofline crossover for this machine:

```
peak_flops / peak_bandwidth ≈ 2400e9 / 97e9 ≈ 25 FLOP per byte
```

Above ~25 FLOP/byte a kernel is compute bound. Matmul at 1024³ has an intensity of ≈171 —
seven times above the line — which is why `labs/lab04` can gain 3x from on-chip reuse while
`labs/lab01` gains nothing at all from any amount of cleverness.

**Sanity check against the spec sheet:** M4 is rated for ~120 GB/s of LPDDR5X bandwidth.
Reaching ~97 GB/s of *usable* read+write traffic is ~80% of nominal — a normal, healthy
figure for a real kernel. If you ever measure far above the nominal number, you have a
bug rather than a discovery; see the next section for a concrete example.

## Things that surprised me while building this repo

These are recorded because they are the kind of thing that only shows up when you
actually measure, and they are the reason several labs end with "and the textbook
ordering did not hold":

1. **A hand-written scalar vector add matches Apple's own kernel** (≈85-88 vs ≈85 GB/s).
   Elementwise work on this hardware is a solved problem; there is nothing to win.
2. **`float4` vectorisation did not reliably beat scalar loads** in my tests, and in some
   configurations measured *worse*. On NVIDIA this is usually a clear win. Measure
   before assuming.
3. **Contended global atomics are cheap here.** `labs/lab03` v1 performs 16.7M atomic
   adds to a *single* address and still reaches ~63 GB/s — better than the classic
   threadgroup tree reduction. On an NVIDIA GPU that kernel is typically an order of
   magnitude off. Apple's memory system handles this very differently.
4. **The tiled transpose wins by only ~1.1-1.5x**, not the ~10x of the CUDA literature.
   The large L2 absorbs much of the scattered-access penalty.
5. **Measurement noise is ±20% between runs, and sustained load throttles.** This is a
   fanless laptop. `gpuk.bench.best_ms` exists for this reason; treat sub-10%
   differences as ties.
6. **A 3x3 convolution stencil is not bandwidth bound here at all.** It costs the same as
   a plain copy of the image, because a warp's nine reads span about four cache lines and
   L1 absorbs the redundancy the textbook says you must tile away. Tiling a 3x3 stencil is
   wasted code; from K≥2 it wins a consistent ~1.6x. See `labs/lab05_convolution.py`.
7. **Staging the convolution filter in on-chip memory measured *slower*.** This is
   unanimous PMPP advice ("put the filter in `__constant__` memory") and it did not
   reproduce at any kernel size, because the compiler already hoists loop-invariant loads
   and the rest is a cached broadcast. The cooperative load plus its barrier cost more
   than it saved.

### Three bugs I wrote while building this repo

All three are preserved here because they produce *confident, wrong* results rather than
errors — the failure mode that costs real time.

**A divergent `simd_sum`.** My first read-bandwidth probe looked like this:

```metal
if (t % 32 == 0) out[t / 32] = simd_sum(acc);   // WRONG
```

A simdgroup shuffle must be reached by all 32 lanes. Inside a divergent branch it reads
garbage from the masked-off lanes — and, worse, the compiler is then free to delete the
loads that were feeding it. The probe reported **167 GB/s on a machine rated for 120**, and
its output did not reproduce the array's sum. The fix is one line:

```metal
float s = simd_sum(acc);           // all 32 lanes
if (t % 32 == 0) out[t / 32] = s;  // only lane 0 stores
```

`bench/m4_peaks.py` now asserts that the probe's reduction matches the array total before
reporting any number. A bandwidth figure from a kernel that read the wrong data is
worthless, and *nonsensically fast is the loudest possible symptom of it*.

**A buffer overrun that page-faulted the GPU.** Casting a `float*` buffer to `float4*`
quadruples the reachable element count, but I had declared the output array with its
original float32 length. The kernel wrote 4x past the end and Metal killed the command
buffer with `kIOGPUCommandBufferCallbackErrorPageFault`. Lesson: when you re-interpret a
buffer as a wider type, the *allocation* is still sized in the base type.

**A barrier inside a partially-dispatched threadgroup.** A 1x1 test image launched
`grid=(1,1)` with `threadgroup=(32,8)`, and the kernel's `threadgroup_barrier` then waited
for 255 threads that were never dispatched. Metal does not reject this; it produces wrong
answers, and only for images smaller than one tile. It took a `(1,1)` test case to find —
which is exactly why `labs/lab05_convolution.py` tests `1x1`, `37x37` and `100x130`
instead of only the 2048x2048 image it benchmarks on. `launch_for()` in that lab floors the
grid at one full threadgroup.

## What this means for how you learn

* Labs 01-03 (elementwise, access patterns, reduction) are **memory bound** on this
  hardware. The lessons are about bytes moved and access patterns, and the wins are
  modest because Apple's memory system is forgiving. Do not conclude "optimisation
  doesn't matter".
* Lab 04 (matmul) is **compute bound** and there the wins are large and unambiguous:
  248 → 410 → 809 GFLOP/s across three versions of the same algorithm, against 1803 for
  Apple's tuned GEMM.
* If you want to *see* the dramatic NVIDIA-style effects (10x from coalescing, 100x from
  tiling), you need an NVIDIA GPU. See `docs/04-pmpp-roadmap.md` for how to get one for
  a few hours without buying anything.

---

*Next: [`03-how-to-measure.md`](03-how-to-measure.md).*
