// PMPP vector add, in Metal Shading Language.
//
// This is a COMPLETE, compilable .metal file — the kind you would build with
// `xcrun metal` in an Xcode project.
//
// MLX generates the signature below for you from the Python call, so when using MLX you
// write only the region between the two markers. `pmpp/run_all.py` extracts exactly that
// region and runs it, which is why this file cannot drift away from a working kernel.
//
//     from gpuk import cuda
//     src = extract_body("pmpp/01_vector_add/kernel.metal")
//     k = mx.fast.metal_kernel(name="vector_add", input_names=["a","b"],
//                              output_names=["c"], source=src)
//     k(inputs=[a, b], output_shapes=[a.shape], output_dtypes=[a.dtype],
//       **cuda(ceil_div(n, 256), 256).kwargs())

#include <metal_stdlib>
using namespace metal;

// CUDA: __global__ void vector_add(const float* a, const float* b, float* c, int n)
//
// Differences visible in the signature alone:
//   * `__global__`            -> `[[kernel]]`
//   * `const float*`          -> `const device float*`   (address space is explicit)
//   * `int n`                 -> a shape buffer, because MLX passes array metadata
//   * no equivalent of threadIdx in the signature -> Metal passes thread attributes
//     as ordinary function parameters
[[kernel]] void vector_add(
    const device float* a                    [[buffer(0)]],
    const device float* b                    [[buffer(1)]],
    device float*       c                    [[buffer(2)]],
    constant uint*      a_shape              [[buffer(3)]],
    uint3               thread_position_in_grid [[thread_position_in_grid]])
{
    // --- MLX body begin ---
    // `thread_position_in_grid` is the pre-computed global thread index — Metal hands
    // you what CUDA makes you write out as blockIdx.x * blockDim.x + threadIdx.x.
    uint i = thread_position_in_grid.x;

    // The same boundary guard. `a_shape[0]` is the array length, injected by MLX
    // because the name appears in the source. In hand-written Metal you would pass `n`
    // yourself.
    if (i < a_shape[0]) {
        c[i] = a[i] + b[i];
    }
    // --- MLX body end ---
}

// Launch geometry, for reference.
//
//   CUDA :  blocks = ceil(n / 256);  vector_add<<<blocks, 256>>>(...)
//   Metal:  dispatchThreads(MTLSize(blocks * 256, 1, 1),   // TOTAL threads
//                           MTLSize(256, 1, 1))
//   MLX  :  grid=(blocks * 256, 1, 1), threadgroup=(256, 1, 1)
//
// Writing grid=(blocks, 1, 1) is the classic mistake: it launches `blocks` threads
// instead of `blocks * 256`, silently computes the wrong thing, and raises no error.
