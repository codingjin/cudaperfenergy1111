#!/bin/bash

# Build all targets
make

# Create output folders
mkdir -p smem db unrolling vec

# Run smem benchmarks
./smem_perf 4096 4096 4096 2>&1 | tee smem/perf
./smem_energy 4096 4096 4096 2>&1 | tee smem/energy

# Run db benchmarks
./db_perf 4096 4096 4096 2>&1 | tee db/perf
./db_energy 4096 4096 4096 2>&1 | tee db/energy

# Run unrolling benchmarks
./i2_perf 4096 4096 4096 2>&1 | tee unrolling/i2_perf
./i2_energy 4096 4096 4096 2>&1 | tee unrolling/i2_energy
./i4_perf 4096 4096 4096 2>&1 | tee unrolling/i4_perf
./i4_energy 4096 4096 4096 2>&1 | tee unrolling/i4_energy

# Run vectorization benchmarks
./vec2_perf 4096 4096 4096 2>&1 | tee vec/vec2_perf
./vec2_energy 4096 4096 4096 2>&1 | tee vec/vec2_energy
./vec4_perf 4096 4096 4096 2>&1 | tee vec/vec4_perf
./vec4_energy 4096 4096 4096 2>&1 | tee vec/vec4_energy
