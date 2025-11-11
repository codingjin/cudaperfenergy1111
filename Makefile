# Makefile for CUDA matrix multiplication programs

# Detect GPU architecture
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | sed 's/\.//')

# Compiler and flags
NVCC := nvcc
NVCC_FLAGS := -arch=sm_$(GPU_ARCH)

TARGETS := naive naive_perf smem smem_perf smem_energy db db_perf db_energy i2 i2_perf i2_energy i2j2 i2j2_perf i2j2_energy i4 i4_perf i4_energy vec2 vec2_perf vec2_energy vec4 vec4_perf vec4_energy


all: $(TARGETS)
	@echo "All targets built successfully with GPU architecture: sm_$(GPU_ARCH)"

# Individual targets
naive: naive.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

naive_perf: naive_perf.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

smem: smem.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

smem_perf: smem_perf.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

smem_energy: smem_energy.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lnvidia-ml

db: db.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

db_perf: db_perf.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

db_energy: db_energy.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lnvidia-ml

i2: i2.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

i2_perf: i2_perf.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

i2_energy: i2_energy.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lnvidia-ml

i2j2: i2j2.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

i2j2_perf: i2j2_perf.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

i2j2_energy: i2j2_energy.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lnvidia-ml

i4: i4.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

i4_perf: i4_perf.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

i4_energy: i4_energy.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lnvidia-ml

vec2: vec2.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

vec2_perf: vec2_perf.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

vec2_energy: vec2_energy.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lnvidia-ml

vec4: vec4.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

vec4_perf: vec4_perf.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

vec4_energy: vec4_energy.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lnvidia-ml

# Clean target
clean:
	rm -f $(TARGETS)

# Phony targets
.PHONY: all clean
