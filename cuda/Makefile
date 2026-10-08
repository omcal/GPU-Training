# Run from any directory with: make -C /path/to/GPU-Training/cuda test
CUDA_HOME ?= /usr/local/cuda-12.9
NVCC = $(CUDA_HOME)/bin/nvcc
ARCH ?= sm_61
NVCCFLAGS ?= -O2 -lineinfo -std=c++17
OUT := ../build/cuda
LAB_SOURCES := $(wildcard lab*.cu)
LABS := $(patsubst %.cu,$(OUT)/%,$(LAB_SOURCES))
PMPP := $(OUT)/pmpp_vector_add $(OUT)/pmpp_tiled_matmul
BINS := $(PMPP) $(LABS)
SANITIZER = $(CUDA_HOME)/bin/compute-sanitizer
.PHONY: all test sanitize racecheck synccheck
all: $(BINS)
$(OUT):
	mkdir -p $@
$(OUT)/lab%: lab%.cu include/common.cuh | $(OUT)
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
$(OUT)/pmpp_vector_add: ../pmpp/01_vector_add/kernel.cu include/common.cuh | $(OUT)
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
$(OUT)/pmpp_tiled_matmul: ../pmpp/02_tiled_matmul/kernel.cu include/common.cuh | $(OUT)
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
test: all
	@set -e; for bin in $(BINS); do "$$bin"; done
sanitize: all
	@set -e; for bin in $(BINS); do $(SANITIZER) --tool memcheck --error-exitcode 1 "$$bin" --test; done
racecheck: all
	@set -e; for bin in $(BINS); do $(SANITIZER) --tool racecheck --error-exitcode 1 "$$bin" --test; done
synccheck: all
	@set -e; for bin in $(BINS); do $(SANITIZER) --tool synccheck --error-exitcode 1 "$$bin" --test; done
