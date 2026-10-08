// Ask Metal itself for the hardware limits of this GPU.
//
//   swift -module-cache-path .swiftcache tools/device_probe.swift
//
// `gpuk.device.metal_limits()` runs this for you. Metal's own MTLDevice is the only
// authority on things like how much threadgroup memory exists — a number you need
// before you design a tiled kernel, and one that no Python API exposes.
//
// Requires the Swift toolchain (Xcode Command Line Tools are enough). This is the
// same Metal framework and the same GPU that MLX drives from Python.

import Metal
import Foundation

guard let d = MTLCreateSystemDefaultDevice() else {
    FileHandle.standardError.write("no Metal device\n".data(using: .utf8)!)
    exit(1)
}

print("name:                         \(d.name)")
print("hasUnifiedMemory:             \(d.hasUnifiedMemory)")
print("maxThreadsPerThreadgroup:     \(d.maxThreadsPerThreadgroup.width)")
print("maxThreadgroupMemoryLength:   \(d.maxThreadgroupMemoryLength)")
print("maxBufferLength:              \(d.maxBufferLength)")
print("recommendedMaxWorkingSetSize: \(d.recommendedMaxWorkingSetSize)")
print("currentAllocatedSize:         \(d.currentAllocatedSize)")
print("supportsFamily(.apple9):      \(d.supportsFamily(.apple9))")
print("supportsFamily(.metal3):      \(d.supportsFamily(.metal3))")
print("supportsFamily(.metal4):      \(d.supportsFamily(.metal4))")
print("argumentBuffersTier:          \(d.argumentBuffersSupport.rawValue)")
