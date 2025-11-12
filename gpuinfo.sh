#!/usr/bin/env bash

# Script to display min/max power limits and GPU core clock rates for GPU 0

set -euo pipefail

GPU_ID=0

echo "=========================================="
echo "GPU Information for Device ${GPU_ID}"
echo "=========================================="
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

# Check if GPU exists
if ! nvidia-smi -i ${GPU_ID} &> /dev/null; then
    echo "❌ GPU ${GPU_ID} not found"
    exit 1
fi

# Get GPU name
GPU_NAME=$(nvidia-smi -i ${GPU_ID} --query-gpu=name --format=csv,noheader)
echo "GPU Name: ${GPU_NAME}"
echo ""

# Get power limits
echo "Power Limits:"
echo "-------------"

power_info=$(nvidia-smi -i ${GPU_ID} -q -d POWER | awk '
    /GPU Power Readings/,/Power Samples/ {
        if ($1=="Min" && $2=="Power" && $3=="Limit") min_power=$5
        if ($1=="Max" && $2=="Power" && $3=="Limit") max_power=$5
        if ($1=="Default" && $2=="Power" && $3=="Limit") default_power=$5
    }
    END {print min_power, max_power, default_power}
')

read MIN_POWER MAX_POWER DEFAULT_POWER <<< "$power_info"

if [[ -z "$MIN_POWER" || -z "$MAX_POWER" ]]; then
    echo "❌ Failed to parse power limits"
    exit 1
fi

echo "  Min Power Limit:     ${MIN_POWER} W"
echo "  Max Power Limit:     ${MAX_POWER} W"
if [[ -n "$DEFAULT_POWER" ]]; then
    echo "  Default Power Limit: ${DEFAULT_POWER} W"
fi
echo ""

# Get GPU core clock rates
echo "GPU Core Clock Rates:"
echo "---------------------"

# Get all supported clock pairs and extract GPU clocks
ALL_GPU_CLOCKS=$(nvidia-smi -i ${GPU_ID} --query-supported-clocks=memory,graphics --format=csv,noheader \
    | tr -d ' ' | awk -F',' '{print $2}' | grep -Eo '[0-9]+' | sort -nu)

if [[ -z "$ALL_GPU_CLOCKS" ]]; then
    echo "❌ Failed to query supported GPU clocks"
    exit 1
fi

# Get min and max from the sorted unique values
MIN_GPU_CLOCK=$(echo "$ALL_GPU_CLOCKS" | head -n1)
MAX_GPU_CLOCK=$(echo "$ALL_GPU_CLOCKS" | tail -n1)

echo "  Min GPU Clock: ${MIN_GPU_CLOCK} MHz"
echo "  Max GPU Clock: ${MAX_GPU_CLOCK} MHz"
echo ""

# Get current settings
echo "Current Settings:"
echo "-----------------"

CURRENT_POWER=$(nvidia-smi -i ${GPU_ID} --query-gpu=power.limit --format=csv,noheader | grep -Eo '[0-9]+\.[0-9]+')
CURRENT_GPU_CLOCK=$(nvidia-smi -i ${GPU_ID} --query-gpu=clocks.gr --format=csv,noheader | grep -Eo '[0-9]+')

if [[ -n "$CURRENT_POWER" ]]; then
    echo "  Current Power Limit: ${CURRENT_POWER} W"
fi

if [[ -n "$CURRENT_GPU_CLOCK" ]]; then
    echo "  Current GPU Clock:   ${CURRENT_GPU_CLOCK} MHz"
fi

echo ""
echo "=========================================="
