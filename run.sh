#!/bin/bash

# Build all targets
make

# Create output folders
mkdir -p smemdir dbdir unrollingdir vecdir

# Run smem benchmarks
./smem_perf 4096 4096 4096 2>&1 | tee smemdir/perf
./smem_energy 4096 4096 4096 2>&1 | tee smemdir/energy

# Run db benchmarks
./db_perf 4096 4096 4096 2>&1 | tee dbdir/perf
./db_energy 4096 4096 4096 2>&1 | tee dbdir/energy

# Run unrolling benchmarks
./i2_perf 4096 4096 4096 2>&1 | tee unrollingdir/i2_perf
./i2_energy 4096 4096 4096 2>&1 | tee unrollingdir/i2_energy
./i4_perf 4096 4096 4096 2>&1 | tee unrollingdir/i4_perf
./i4_energy 4096 4096 4096 2>&1 | tee unrollingdir/i4_energy

# Run vectorization benchmarks
./vec2_perf 4096 4096 4096 2>&1 | tee vecdir/vec2_perf
./vec2_energy 4096 4096 4096 2>&1 | tee vecdir/vec2_energy
./vec4_perf 4096 4096 4096 2>&1 | tee vecdir/vec4_perf
./vec4_energy 4096 4096 4096 2>&1 | tee vecdir/vec4_energy
