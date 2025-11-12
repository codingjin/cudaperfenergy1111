#!/usr/bin/env bash

# Exit on error, but allow some commands to fail gracefully
set -euxo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# GPU to use (default: 0)
GPU_ID=0

# Matrix dimensions (can be overridden by command line args)
M=${1:-4096}
N=${2:-4096}
K=${3:-4096}

echo "=========================================="
echo "CUDA Matrix Multiplication Benchmark Suite"
echo "Running all experiments across 4 GPU modes"
echo "Matrix dimensions: ${M}x${N}x${K}"
echo "=========================================="
echo ""

# Check for sudo access upfront
if ! sudo -n true 2>/dev/null; then
    echo "⚠️  This script requires sudo access for GPU configuration."
    echo "   You may be prompted for your password."
    sudo -v || { echo "❌ Failed to obtain sudo privileges"; exit 1; }
fi

# Validate nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

# Check if GPU exists
if ! nvidia-smi -i ${GPU_ID} &> /dev/null; then
    echo "❌ GPU ${GPU_ID} not found"
    exit 1
fi

# Store original GPU settings for cleanup
ORIGINAL_POWER_LIMIT=$(nvidia-smi -i ${GPU_ID} -q -d POWER | awk '/Default Power Limit/ {print $5; exit}')
if [[ -z "$ORIGINAL_POWER_LIMIT" ]]; then
    echo "⚠️  Warning: Could not determine default power limit"
    ORIGINAL_POWER_LIMIT=""
fi

# Cleanup function - will be called on exit
cleanup() {
    local exit_code=$?
    echo ""
    echo "=========================================="
    echo "Cleaning up..."
    echo "=========================================="

    # Reset GPU clocks
    echo "→ Resetting GPU clock locks..."
    sudo nvidia-smi -i ${GPU_ID} -rgc || echo "⚠️  Failed to reset GPU clocks"

    # Reset power limit if we have the original value
    if [[ -n "$ORIGINAL_POWER_LIMIT" ]]; then
        echo "→ Resetting power limit to ${ORIGINAL_POWER_LIMIT} W..."
        sudo nvidia-smi -i ${GPU_ID} -pl ${ORIGINAL_POWER_LIMIT} || echo "⚠️  Failed to reset power limit"
    fi

    echo "✅ Cleanup completed"

    if [ $exit_code -ne 0 ]; then
        echo "❌ Script exited with errors (exit code: $exit_code)"
    fi
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Function to configure GPU settings
configure_gpu() {
    local MODE=$1

    echo "Configuring GPU for mode ${MODE}..."

    # Enable persistence mode
    sudo nvidia-smi -i ${GPU_ID} -pm 1 || {
        echo "❌ Failed to enable persistence mode"
        return 1
    }

    # Get MIN/MAX power limits
    local power_info
    power_info=$(nvidia-smi -i ${GPU_ID} -q -d POWER | awk '
        /GPU Power Readings/,/Power Samples/ {
            if ($1=="Min" && $2=="Power" && $3=="Limit") p1=$5
            if ($1=="Max" && $2=="Power" && $3=="Limit") p2=$5
        }
        END {print p1, p2}
    ')

    read MIN_POWER MAX_POWER <<< "$power_info"

    if [[ -z "$MIN_POWER" || -z "$MAX_POWER" ]]; then
        echo "❌ Failed to parse power limits from nvidia-smi"
        echo "   Raw output: $power_info"
        return 1
    fi

    # Validate power limits are numeric
    if ! [[ "$MIN_POWER" =~ ^[0-9]+\.?[0-9]*$ ]] || ! [[ "$MAX_POWER" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "❌ Invalid power limit values: MIN=$MIN_POWER, MAX=$MAX_POWER"
        return 1
    fi

    # Set power limit based on mode
    local POWER_LIMIT
    if [[ "$MODE" == "1" || "$MODE" == "2" ]]; then
        POWER_LIMIT=$MAX_POWER
    else
        POWER_LIMIT=$MIN_POWER
    fi

    echo "→ Setting GPU${GPU_ID} power limit to ${POWER_LIMIT} W"
    sudo nvidia-smi -i ${GPU_ID} -pl ${POWER_LIMIT} || {
        echo "❌ Failed to set power limit"
        return 1
    }

    # Determine GPU core clock
    local GPU_CLOCK
    if [[ "$MODE" == "1" || "$MODE" == "3" ]]; then
        # Max GPU clock
        GPU_CLOCK=$(nvidia-smi -i ${GPU_ID} --query-supported-clocks=memory,graphics --format=csv,noheader \
            | tr -d ' ' | sort -nr -t',' -k2 | head -n1 | awk -F',' '{print $2}' | grep -Eo '[0-9]+')
    else
        # Min GPU clock
        GPU_CLOCK=$(nvidia-smi -i ${GPU_ID} --query-supported-clocks=memory,graphics --format=csv,noheader \
            | tr -d ' ' | sort -n -t',' -k2 | head -n1 | awk -F',' '{print $2}' | grep -Eo '[0-9]+')
    fi

    if [[ -z "$GPU_CLOCK" ]] || ! [[ "$GPU_CLOCK" =~ ^[0-9]+$ ]]; then
        echo "❌ Failed to determine GPU clock frequency"
        return 1
    fi

    echo "→ Locking GPU core clock to ${GPU_CLOCK} MHz"
    sudo nvidia-smi -i ${GPU_ID} --lock-gpu-clocks=${GPU_CLOCK},${GPU_CLOCK} || {
        echo "❌ Failed to lock GPU clocks"
        return 1
    }

    # Disable other GPUs
    local NUM_GPUS
    NUM_GPUS=$(nvidia-smi -L | wc -l)

    if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
        echo "⚠️  Warning: Could not determine number of GPUs"
    elif [ "$NUM_GPUS" -gt 1 ]; then
        echo "→ Disabling other GPUs (total: ${NUM_GPUS})..."
        for ((i=1; i<$NUM_GPUS; i++)); do
            local BUS_ID
            BUS_ID=$(nvidia-smi -i $i -q | awk '/Bus Id/ {print $4; exit}')
            if [ -n "$BUS_ID" ]; then
                echo "  Unbinding GPU $i (Bus $BUS_ID)..."
                sudo sh -c "echo $BUS_ID > /sys/bus/pci/drivers/nvidia/unbind" 2>/dev/null || \
                    echo "  ⚠️  Failed to unbind GPU $i (may already be unbound)"
            fi
        done
    fi

    echo "✅ GPU configured in mode $MODE (Power: ${POWER_LIMIT}W, Clock: ${GPU_CLOCK}MHz)"
    echo ""

    return 0
}

# Function to run benchmarks for a specific mode
run_benchmarks() {
    local MODE=$1
    local SUFFIX=""

    case $MODE in
        1) SUFFIX="pmax_cmax" ;;
        2) SUFFIX="pmax_cmin" ;;
        3) SUFFIX="pmin_cmax" ;;
        4) SUFFIX="pmin_cmin" ;;
        *) echo "❌ Invalid mode: $MODE"; return 1 ;;
    esac

    echo "=========================================="
    echo "Running benchmarks for Mode ${MODE} (${SUFFIX})"
    echo "=========================================="
    echo ""

    # Set CUDA_VISIBLE_DEVICES for all benchmark runs
    export CUDA_VISIBLE_DEVICES=0

    # List of all benchmarks to run: "binary args output_file"
    local benchmarks=(
        "smem_perf:smemdir/perf_${SUFFIX}"
        "smem_energy:smemdir/energy_${SUFFIX}"
        "db_perf:dbdir/perf_${SUFFIX}"
        "db_energy:dbdir/energy_${SUFFIX}"
        "i2_perf:unrollingdir/i2_perf_${SUFFIX}"
        "i2_energy:unrollingdir/i2_energy_${SUFFIX}"
        "i4_perf:unrollingdir/i4_perf_${SUFFIX}"
        "i4_energy:unrollingdir/i4_energy_${SUFFIX}"
        "vec2_perf:vecdir/vec2_perf_${SUFFIX}"
        "vec2_energy:vecdir/vec2_energy_${SUFFIX}"
        "vec4_perf:vecdir/vec4_perf_${SUFFIX}"
        "vec4_energy:vecdir/vec4_energy_${SUFFIX}"
    )

    for benchmark_spec in "${benchmarks[@]}"; do
        IFS=':' read -r binary output_file <<< "$benchmark_spec"

        if [[ ! -x "./$binary" ]]; then
            echo "❌ Binary ./$binary not found or not executable. Skipping..."
            continue
        fi

        echo "→ Running $binary..."
        if ! ./$binary $M $N $K 2>&1 | tee "$output_file"; then
            echo "⚠️  Warning: $binary failed or returned non-zero exit code"
        fi
    done

    echo ""
    echo "✅ Mode ${MODE} (${SUFFIX}) benchmarks completed"
    echo ""

    return 0
}

# Main execution
echo "Step 1: Building all targets..."
if ! make clean; then
    echo "⚠️  Warning: make clean failed"
fi

if ! make; then
    echo "❌ Build failed"
    exit 1
fi
echo "✅ Build completed"
echo ""

echo "Step 2: Creating output directories..."
mkdir -p smemdir dbdir unrollingdir vecdir
echo "✅ Directories created"
echo ""

echo "Step 3: Verifying all binaries exist..."
MISSING_BINARIES=0
for binary in smem_perf smem_energy db_perf db_energy i2_perf i2_energy i4_perf i4_energy vec2_perf vec2_energy vec4_perf vec4_energy; do
    if [[ ! -x "./$binary" ]]; then
        echo "  ❌ Missing: $binary"
        MISSING_BINARIES=1
    fi
done

if [ $MISSING_BINARIES -eq 1 ]; then
    echo "❌ Some binaries are missing. Please check the build output."
    exit 1
fi
echo "✅ All binaries present"
echo ""

# Run benchmarks for all 4 modes
for MODE in 1 2 3 4; do
    if ! configure_gpu $MODE; then
        echo "❌ Failed to configure GPU for mode $MODE"
        exit 1
    fi

    if ! run_benchmarks $MODE; then
        echo "⚠️  Some benchmarks failed in mode $MODE, but continuing..."
    fi

    # Add a cooldown period between modes
    if [ $MODE -lt 4 ]; then
        echo "Cooling down for 10 seconds before next mode..."
        sleep 10
        echo ""
    fi
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - smemdir/perf_* and smemdir/energy_*"
echo "  - dbdir/perf_* and dbdir/energy_*"
echo "  - unrollingdir/*_perf_* and unrollingdir/*_energy_*"
echo "  - vecdir/*_perf_* and vecdir/*_energy_*"
echo ""
echo "Suffix meanings:"
echo "  - pmax_cmax: power max, clock max (Mode 1)"
echo "  - pmax_cmin: power max, clock min (Mode 2)"
echo "  - pmin_cmax: power min, clock max (Mode 3)"
echo "  - pmin_cmin: power min, clock min (Mode 4)"
echo ""
echo "Matrix dimensions used: ${M}x${N}x${K}"
