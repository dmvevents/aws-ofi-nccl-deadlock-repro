#!/bin/bash
#SBATCH --job-name=moe17b-moe
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=moe17b-%j.out
#SBATCH --error=moe17b-%j.err

#
# SLURM script for MoE17B 17B MoE Training
#
# This script runs the MoE17B MoE model training to stress-test
# the NCCL/EFA communication path, particularly All-to-All operations.
#
# Usage:
#   # Standard training (with fixed aws-ofi-nccl)
#   sbatch slurm/train_moe.sh
#
#   # With vulnerable library (to test deadlock)
#   LIBRARY_MODE=vulnerable sbatch slurm/train_moe.sh
#
# Requirements:
#   - NeMo container (nvcr.io/nvidia/nemo:25.09.00 or later)
#   - AWS EFA networking
#   - Pyxis for container support
#

set -e

echo "============================================================"
echo "MoE17B 17B MoE Training"
echo "============================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Nodes:     $SLURM_NNODES"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Date:      $(date)"
echo ""

# =============================================================================
# CONFIGURATION
# =============================================================================

# Container image (update this path)
CONTAINER_IMAGE=${CONTAINER_IMAGE:-"/path/to/nemo-container.sqsh"}

# Library mode: "fixed" (default) or "vulnerable"
LIBRARY_MODE=${LIBRARY_MODE:-fixed}

# Training configuration
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
MAX_STEPS=${MAX_STEPS:-1000}
RESULTS_DIR=${RESULTS_DIR:-"/tmp/moe17b_results_$SLURM_JOB_ID"}

echo "Configuration:"
echo "  CONTAINER_IMAGE: $CONTAINER_IMAGE"
echo "  LIBRARY_MODE:    $LIBRARY_MODE"
echo "  NUM_NODES:       $NUM_NODES"
echo "  GPUS_PER_NODE:   $GPUS_PER_NODE"
echo "  MAX_STEPS:       $MAX_STEPS"
echo "  RESULTS_DIR:     $RESULTS_DIR"
echo ""

# =============================================================================
# CRITICAL: AWS OFI NCCL Configuration
# =============================================================================

# Load AWS OFI NCCL plugin for EFA RDMA
export LD_LIBRARY_PATH=/opt/amazon/aws-ofi-nccl/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH

# Tell NCCL to use AWS Libfabric plugin
export NCCL_NET="AWS Libfabric"

# EFA optimizations
export FI_PROVIDER=efa
export FI_EFA_USE_HUGE_PAGE=0  # Fix for fork/GC memory issues

# NCCL settings
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_TIMEOUT=1800
export NCCL_NVLS_ENABLE=0  # Disable NVLS - fixes NVLink peer access errors

# PyTorch distributed timeout (must match NCCL_TIMEOUT)
export TORCH_DISTRIBUTED_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Triton cache on local disk (not shared filesystem)
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}_${SLURM_JOB_ID}

# Training configuration
export NUM_NODES
export GPUS_PER_NODE
export MAX_STEPS
export RESULTS_DIR

echo "Environment:"
echo "  LD_LIBRARY_PATH: $(echo $LD_LIBRARY_PATH | cut -d: -f1-2)..."
echo "  NCCL_NET:        $NCCL_NET"
echo "  FI_PROVIDER:     $FI_PROVIDER"
echo "  NCCL_DEBUG:      $NCCL_DEBUG"
echo "  NCCL_TIMEOUT:    $NCCL_TIMEOUT"
echo ""

# =============================================================================
# LIBRARY MODE SELECTION
# =============================================================================

if [ "$LIBRARY_MODE" = "vulnerable" ]; then
    echo "WARNING: Using VULNERABLE aws-ofi-nccl library"
    echo "This may cause deadlocks under memory registration stress!"
    echo ""
    # Override library path to use vulnerable version
    export LD_LIBRARY_PATH=/opt/aws-ofi-nccl-1.14.0-vulnerable/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
fi

# =============================================================================
# DIAGNOSTIC: Run on rank 0 only
# =============================================================================

echo "Running diagnostics on rank 0..."
srun --ntasks=1 --nodes=1 --export=ALL \
    --container-image="$CONTAINER_IMAGE" \
    bash -c '
export LD_LIBRARY_PATH=/opt/amazon/aws-ofi-nccl/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export NCCL_NET="AWS Libfabric"
echo "=== DIAGNOSTIC INFO ==="
echo "AWS OFI NCCL location:"
ls -la /opt/amazon/aws-ofi-nccl/lib/ 2>/dev/null || echo "  Not found"
echo ""
echo "EFA libraries:"
ls -la /opt/amazon/efa/lib/libfabric.so* 2>/dev/null | head -3 || echo "  Not found"
echo ""
echo "EFA devices:"
/opt/amazon/efa/bin/fi_info -p efa 2>/dev/null | head -10 || echo "  fi_info failed"
echo ""
echo "aws-ofi-nccl version:"
strings /opt/amazon/aws-ofi-nccl/lib/libnccl-net.so 2>/dev/null | grep -i "aws-ofi-nccl" | head -1 || echo "  Unknown"
echo "=== END DIAGNOSTIC ==="
'

echo ""
echo "============================================================"
echo "Starting training..."
echo "============================================================"
echo ""

# =============================================================================
# RUN TRAINING
# =============================================================================

srun --export=ALL \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts='/tmp:/tmp' \
    --gres=gpu:8 \
    bash -c '
# Set library path inside container
if [ "$LIBRARY_MODE" = "vulnerable" ]; then
    export LD_LIBRARY_PATH=/opt/aws-ofi-nccl-1.14.0-vulnerable/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=/opt/amazon/aws-ofi-nccl/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
fi
export NCCL_NET="AWS Libfabric"
export TORCH_DISTRIBUTED_TIMEOUT=1800

# Run training script
python3 /path/to/nemo/train_moe.py
'

echo ""
echo "============================================================"
echo "Training completed"
echo "============================================================"
