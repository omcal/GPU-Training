# pmpp/ — the same kernels in CUDA and Metal, side by side

The point of this directory is **translation confidence**. For each canonical PMPP
kernel there is a `kernel.cu` (what the book writes), a `kernel.metal` (the equivalent
Metal Shading Language), and a `notes.md` that walks the translation line by line and
calls out every place where the two diverge.

```
01_vector_add/     kernel.cu  kernel.metal  notes.md
02_tiled_matmul/   kernel.cu  kernel.metal  notes.md
run_all.py         runs the Metal versions and checks them against numpy
```

On the Mac (Metal):

```bash
python pmpp/run_all.py
```

On Ubuntu (CUDA), from the project root:

```bash
make -C cuda all
./build/cuda/pmpp_vector_add
./build/cuda/pmpp_tiled_matmul
```

See [`../cuda/README.md`](../cuda/README.md) for all six native CUDA labs.

## How the `.metal` files relate to MLX

Each `kernel.metal` is **complete, valid Metal Shading Language** — the kind of file you
would compile with `xcrun metal` in an Xcode project.

MLX generates that function signature (buffers, `[[buffer(n)]]`, the thread attributes)
from the Python call, so when using MLX you write only the *body*. Those bodies are
marked in the `.metal` files:

```metal
[[kernel]] void vector_add(...) {      // <- MLX generates this whole line
    // --- MLX body begin ---
    ...                                // <- this is what you pass as source=
    // --- MLX body end ---
}
```

`pmpp/run_all.py` extracts exactly that region and feeds it to
`mx.fast.metal_kernel`, so the file you read is the file that runs. No duplication, and
no chance of the reference drifting away from a working kernel.

## Why bother, if the labs already have Metal kernels?

Because reading CUDA and writing Metal is a skill you build by seeing them adjacent, and
because PMPP is where the *theory* lives. The labs teach you to write Metal; this
directory teaches you to read the book.

See [`../docs/01-cuda-to-metal.md`](../docs/01-cuda-to-metal.md) for the full translation
table and the one gotcha that matters most (`grid` counts threads, not blocks).
