#!/bin/bash
#SBATCH --job-name=ofi-nccl-deadlock-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=deadlock-test-%j.out
#SBATCH --error=deadlock-test-%j.err

#
# SLURM script for aws-ofi-nccl Deadlock Reproduction Test
#
# Usage:
#   # Vulnerable test (should deadlock)
#   LIBRARY_MODE=vulnerable sbatch run_test.sh
#
#   # Control test (should complete)
#   LIBRARY_MODE=fixed sbatch run_test.sh
#
# Prerequisites:
#   - Enroot/Pyxis for container support, OR
#   - Container image converted to squashfs
#

set -e

echo "============================================================"
echo "AWS OFI NCCL Deadlock Reproduction Test (SLURM)"
echo "============================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Nodes:     $SLURM_NNODES"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Date:      $(date)"
echo ""

# ============================================================
# Configuration
# ============================================================

# Library mode (vulnerable or fixed)
LIBRARY_MODE=${LIBRARY_MODE:-vulnerable}

# Error injection settings
MR_ERROR_RATE=${MR_ERROR_RATE:-50}
MR_ERROR_START=${MR_ERROR_START:-500}

# Test settings
ITERATIONS=${ITERATIONS:-1000}
GPUS_PER_NODE=8

# Container image (update this path)
CONTAINER_IMAGE=${CONTAINER_IMAGE:-"/path/to/aws-ofi-nccl-deadlock-test.sqsh"}

echo "Configuration:"
echo "  LIBRARY_MODE:   $LIBRARY_MODE"
echo "  MR_ERROR_RATE:  $MR_ERROR_RATE"
echo "  MR_ERROR_START: $MR_ERROR_START"
echo "  ITERATIONS:     $ITERATIONS"
echo "  CONTAINER:      $CONTAINER_IMAGE"
echo ""

# ============================================================
# Environment setup
# ============================================================

# EFA settings
export FI_PROVIDER=efa
export FI_EFA_USE_HUGE_PAGE=0

# NCCL settings
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-300}
export NCCL_NVLS_ENABLE=0

# PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Triton cache (must be local, not shared filesystem)
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}_${SLURM_JOB_ID}

# Test configuration
export LIBRARY_MODE
export MR_ERROR_RATE
export MR_ERROR_START
export ITERATIONS
export GPUS_PER_NODE
export NNODES=$SLURM_NNODES

echo "Environment:"
echo "  FI_PROVIDER:    $FI_PROVIDER"
echo "  NCCL_DEBUG:     $NCCL_DEBUG"
echo "  NCCL_TIMEOUT:   $NCCL_TIMEOUT"
echo ""
echo "============================================================"
echo ""

# ============================================================
# Run the test
# ============================================================

# Option 1: Using Pyxis (container support for SLURM)
if command -v srun &> /dev/null && [ -f "$CONTAINER_IMAGE" ]; then
    echo "Running with container via Pyxis..."
    srun --export=ALL \
         --container-image="$CONTAINER_IMAGE" \
         --container-mounts="/dev/infiniband:/dev/infiniband,/dev/gdrdrv:/dev/gdrdrv" \
         --gres=gpu:8 \
         bash -c '
            export LD_LIBRARY_PATH=/opt/amazon/aws-ofi-nccl/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
            if [ "$LIBRARY_MODE" = "vulnerable" ]; then
                export LD_LIBRARY_PATH=/opt/aws-ofi-nccl-1.14.0-vulnerable/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
            fi
            if [ "$MR_ERROR_RATE" != "0" ]; then
                export LD_PRELOAD=/opt/inject_mr_errors.so
            fi
            python3 /opt/tests/moe_stress_test.py
         '

# Option 2: Native execution (libraries installed on nodes)
else
    echo "Running natively (no container)..."
    echo "WARNING: Requires aws-ofi-nccl installed at expected paths"

    # Determine library path based on mode
    if [ "$LIBRARY_MODE" = "vulnerable" ]; then
        LIB_PATH="/opt/aws-ofi-nccl-1.14.0-vulnerable/lib"
    else
        LIB_PATH="/opt/amazon/aws-ofi-nccl/lib"
    fi

    export LD_LIBRARY_PATH=$LIB_PATH:/opt/amazon/efa/lib:$LD_LIBRARY_PATH

    # Enable error injection if requested
    if [ "$MR_ERROR_RATE" != "0" ] && [ -f /opt/inject_mr_errors.so ]; then
        export LD_PRELOAD=/opt/inject_mr_errors.so
    fi

    # Get master node
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    export MASTER_ADDR
    export MASTER_PORT=29500

    srun --export=ALL \
         torchrun \
             --nnodes=$SLURM_NNODES \
             --nproc_per_node=$GPUS_PER_NODE \
             --node_rank=$SLURM_NODEID \
             --master_addr=$MASTER_ADDR \
             --master_port=$MASTER_PORT \
             /opt/tests/moe_stress_test.py
fi

echo ""
echo "============================================================"
echo "Test completed"
echo "============================================================"
