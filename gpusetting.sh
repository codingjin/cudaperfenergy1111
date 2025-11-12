#!/usr/bin/env bash
# ============================================================
# Configure NVIDIA GPU0 in 4 modes and disable all other GPUs
# Modes:
#   1: power max, clock max
#   2: power max, clock min
#   3: power min, clock max
#   4: power min, clock min
# ============================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <mode_number>"
    exit 1
fi

MODE=$1
GPU_ID=0
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPU(s)."

# -----------------------------
# Configure GPU0 (as before)
# -----------------------------
sudo nvidia-smi -i ${GPU_ID} -pm 1

# Power limits
MIN_POWER=$(nvidia-smi -i ${GPU_ID} -q -d POWER | grep "Min Power Limit" | awk '{print $5}')
MAX_POWER=$(nvidia-smi -i ${GPU_ID} -q -d POWER | grep "Max Power Limit" | awk '{print $5}')

if [[ "$MODE" == "1" || "$MODE" == "2" ]]; then
    POWER_LIMIT=$MAX_POWER
else
    POWER_LIMIT=$MIN_POWER
fi

sudo nvidia-smi -i ${GPU_ID} -pl ${POWER_LIMIT}

# Clock settings
if [[ "$MODE" == "1" || "$MODE" == "3" ]]; then
    GPU_CLOCK=$(nvidia-smi -i ${GPU_ID} --query-supported-clocks=memory,graphics --format=csv,noheader \
                | sort -nr -t',' -k2 | head -n1 | awk -F',' '{print $2}' | xargs)
    MEM_CLOCK=$(nvidia-smi -i ${GPU_ID} --query-supported-clocks=memory,graphics --format=csv,noheader \
                | sort -nr -t',' -k2 | head -n1 | awk -F',' '{print $1}' | xargs)
else
    GPU_CLOCK=$(nvidia-smi -i ${GPU_ID} --query-supported-clocks=memory,graphics --format=csv,noheader \
                | sort -n -t',' -k2 | head -n1 | awk -F',' '{print $2}' | xargs)
    MEM_CLOCK=$(nvidia-smi -i ${GPU_ID} --query-supported-clocks=memory,graphics --format=csv,noheader \
                | sort -n -t',' -k2 | head -n1 | awk -F',' '{print $1}' | xargs)
fi

sudo nvidia-smi -i ${GPU_ID} --lock-gpu-clocks=${GPU_CLOCK},${GPU_CLOCK}
sudo nvidia-smi -i ${GPU_ID} --lock-memory-clocks=${MEM_CLOCK},${MEM_CLOCK}

# -----------------------------
# Disable other GPUs
# -----------------------------
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Disabling other GPUs..."
    for ((i=1; i<$NUM_GPUS; i++)); do
        # Get PCI Bus ID
        BUS_ID=$(nvidia-smi -i $i -q | grep "Bus Id" | awk '{print $4}')
        echo "→ Unbinding GPU $i (Bus $BUS_ID) from driver"
        sudo sh -c "echo $BUS_ID > /sys/bus/pci/drivers/nvidia/unbind"
    done
fi

# -----------------------------
# CUDA visibility
# -----------------------------
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES=0"
echo "✅ GPU0 configured in mode $MODE; other GPUs disabled."
echo "Verify with: nvidia-smi -q -d POWER,CLOCK"
