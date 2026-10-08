"""Launch geometry: CUDA semantics -> Metal `dispatchThreads` semantics.

This is **the** API difference that trips up every CUDA programmer on Metal, so it
gets its own module.

CUDA
----
``kernel<<<gridDim, blockDim>>>(...)``
``gridDim`` counts **blocks** (threadgroups). A kernel with 16 blocks x 256 threads
runs 4096 threads. Inside the kernel ``blockIdx.x in [0,16)`` and ``threadIdx.x in [0,256)``.

Metal / MLX
-----------
``dispatchThreads(threadsPerGrid, threadsPerThreadgroup)``
``threadsPerGrid`` is the **total number of threads**, not a number of blocks.
MLX exposes it as ``grid=`` and ``threadgroup=``. To get the CUDA behaviour:

    grid       = (blocks * threads, 1, 1)      # <-- TOTAL threads
    threadgroup= (threads, 1, 1)

Getting this wrong does not error. It silently launches the wrong number of threads
and your result is quietly incorrect. Always sanity-check a new kernel near a
boundary sizing (e.g. 3000 elements with 256 threads).

Inside the MSL kernel the names differ too::

    blockIdx.x   -> threadgroup_position_in_grid.x
    threadIdx.x  -> thread_index_in_threadgroup
    blockDim.x   -> threads_per_threadgroup.x
    gridDim.x    -> threadgroups_per_grid.x
    global tid   -> thread_position_in_grid.x   (free, most kernels use this)

One more Metal constraint: every ``threadgroup`` dimension must be <= the matching
``grid`` dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_THREADS = 256


def cdiv(a: int, b: int) -> int:
    """Ceiling division — the standard way to size a grid in CUDA."""
    return -(-a // b)


@dataclass(frozen=True)
class Launch:
    """A resolved launch geometry, ready to splat into an MLX kernel call."""

    blocks: tuple[int, int, int]
    threads: tuple[int, int, int]

    @property
    def total_threads(self) -> int:
        return self.blocks[0] * self.blocks[1] * self.blocks[2] * (
            self.threads[0] * self.threads[1] * self.threads[2]
        )

    def kwargs(self) -> dict[str, tuple[int, int, int]]:
        """Return ``{"grid": ..., "threadgroup": ...}`` for a kernel call."""
        return {
            "grid": (
                self.blocks[0] * self.threads[0],
                self.blocks[1] * self.threads[1],
                self.blocks[2] * self.threads[2],
            ),
            "threadgroup": self.threads,
        }

    def __str__(self) -> str:  # pragma: no cover - display only
        b = "x".join(str(x) for x in self.blocks)
        t = "x".join(str(x) for x in self.threads)
        return f"<<<{b} blocks, {t} threads>>> = {self.total_threads} threads"


def cuda(blocks: int | tuple[int, ...], threads: int | tuple[int, ...] = DEFAULT_THREADS) -> Launch:
    """Build a launch that means exactly what ``<<<blocks, threads>>>`` means in CUDA.

    >>> cuda(16, 256).total_threads
    4096
    >>> cuda(16, 256).kwargs()["grid"]
    (4096, 1, 1)

    Negative or zero sizes are a programming error, raised loudly rather than
    launching a silently empty grid.
    """
    if isinstance(blocks, int):
        blocks = (blocks,)
    if isinstance(threads, int):
        threads = (threads,)
    if any(b <= 0 for b in blocks):
        raise ValueError(f"blocks must be positive, got {blocks}")
    if any(t <= 0 for t in threads):
        raise ValueError(f"threads must be positive, got {threads}")

    def pad3(dims: tuple[int, ...]) -> tuple[int, int, int]:
        if len(dims) > 3:
            raise ValueError(f"at most 3 dimensions supported, got {dims}")
        return tuple(list(dims) + [1] * (3 - len(dims)))  # type: ignore[return-value]

    b3, t3 = pad3(blocks), pad3(threads)
    if t3[0] * t3[1] * t3[2] > 1024:
        raise ValueError(
            f"{t3[0]}x{t3[1]}x{t3[2]} threads per threadgroup exceeds the Apple GPU "
            "limit of 1024 (verified via tools/device_probe.swift)"
        )
    return Launch(blocks=b3, threads=t3)


def blocks_for(n: int, threads: int = DEFAULT_THREADS) -> Launch:
    """The classic CUDA sizing idiom: one thread per element, rounded up.

    Use this for elementwise kernels. For anything bigger, prefer a grid-stride
    loop (see ``labs/lab02_coalescing.py``) so the grid stays independent of ``n``.
    """
    return cuda(cdiv(n, threads), threads)
