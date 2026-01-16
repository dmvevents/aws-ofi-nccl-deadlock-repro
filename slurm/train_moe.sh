#!/bin/bash
#SBATCH --nodes=62
#SBATCH --job-name=moe17b_train
#SBATCH --partition=p5-queue
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --error=/fsx/results/logs/v3.%j.err
#SBATCH --output=/fsx/results/logs/v3.%j.out

# =============================================================================
# MINIMAL CONFIG - Based on working job 25 settings
# Only added: FI_EFA_USE_HUGE_PAGE=0 and TRITON_CACHE_DIR fix
# =============================================================================

# CRITICAL: Load AWS OFI NCCL plugin for EFA RDMA
export LD_LIBRARY_PATH=/opt/amazon/aws-ofi-nccl/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH

# CRITICAL: Tell NCCL to use AWS Libfabric plugin
# This is REQUIRED for NCCL to find the aws-ofi-nccl plugin
export NCCL_NET="AWS Libfabric"

# EFA optimizations for p5.48xlarge (same as working job)
export FI_PROVIDER=efa
export FI_EFA_USE_HUGE_PAGE=0         # Fix for fork/GC memory issues

# NCCL settings (same as working job)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_TIMEOUT=1800
export NCCL_NVLS_ENABLE=0             # Disable NVLS - fixes NVLink peer access errors

# =============================================================================
# CRITICAL FIX: PyTorch Distributed Timeout
# The ALLGATHER timeout during checkpoint load was caused by PyTorch's
# ProcessGroup watchdog timeout (default 600s = 10 min), NOT NCCL timeout.
# With 512 GPUs resharding a checkpoint from 31→64 nodes on FSx, this takes
# longer than 10 minutes. Set to match NCCL_TIMEOUT.
# =============================================================================
export TORCH_DISTRIBUTED_TIMEOUT=1800     # 30 min - MUST match NCCL_TIMEOUT
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # Better error handling

# PyTorch (same as working job)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CRITICAL FIX: Triton cache on local disk (not shared FSx)
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}_${SLURM_JOB_ID}

# Cache directories (same as working job)
export HF_HOME="/fsx/training/cache/huggingface"
export TRANSFORMERS_CACHE="/fsx/training/cache/huggingface"
export HF_DATASETS_CACHE="/fsx/training/cache/huggingface/datasets"
export NEMO_CACHE_DIR="/fsx/training/cache/nemo"
export TORCH_HOME="/fsx/training/cache/torch"

# WandB API Key (set your own key here)
# export WANDB_API_KEY=your_key_here

# =============================================================================
# DIAGNOSTIC: Run on rank 0 only to check EFA/RDMA setup
# =============================================================================
srun --ntasks=1 --nodes=1 --export=ALL \
        --container-image='/fsx/containers/nemo-container.sqsh' \
        --container-mounts='/fsx:/fsx' \
        bash -c '
export LD_LIBRARY_PATH=/opt/amazon/aws-ofi-nccl/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export NCCL_NET="AWS Libfabric"
echo "========================================="
echo "DIAGNOSTIC INFO (Rank 0 only)"
echo "========================================="
echo ""
echo "1. Checking AWS OFI NCCL plugin locations:"
echo "   Looking in /opt/amazon/aws-ofi-nccl/lib/:"
ls -la /opt/amazon/aws-ofi-nccl/lib/ 2>/dev/null || echo "   ERROR: /opt/amazon/aws-ofi-nccl/lib/ not found!"
echo ""
echo "   Looking for libnccl-net*.so anywhere in /opt:"
find /opt -name "libnccl-net*.so" -o -name "libnccl-ofi*.so" 2>/dev/null
echo ""
echo "2. Checking EFA libraries:"
ls -la /opt/amazon/efa/lib/libfabric.so* 2>/dev/null | head -3 || echo "   ERROR: EFA libs not found!"
echo ""
echo "3. Environment variables:"
echo "   LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "   NCCL_NET=$NCCL_NET"
echo "   FI_PROVIDER=$FI_PROVIDER"
echo ""
echo "4. EFA devices (fi_info -p efa):"
/opt/amazon/efa/bin/fi_info -p efa 2>/dev/null | head -15 || echo "   ERROR: fi_info failed or no EFA devices"
echo ""
echo "5. Check for NCCL plugin in standard paths:"
for path in /opt/amazon/aws-ofi-nccl/lib /opt/amazon/ofi-nccl/lib /usr/lib/x86_64-linux-gnu; do
    if [ -f "$path/libnccl-net.so" ]; then
        echo "   FOUND: $path/libnccl-net.so"
        ls -la "$path/libnccl-net.so"
    fi
done
echo ""
echo "6. ldd check on plugin (if found):"
PLUGIN=$(find /opt -name "libnccl-net.so" 2>/dev/null | head -1)
if [ -n "$PLUGIN" ]; then
    echo "   Checking: $PLUGIN"
    ldd "$PLUGIN" 2>/dev/null | grep -E "not found|libfabric|efa"
else
    echo "   ERROR: No libnccl-net.so found in /opt"
fi
echo ""
echo "7. aws-ofi-nccl version (if available):"
strings /opt/amazon/aws-ofi-nccl/lib/libnccl-net.so 2>/dev/null | grep -i "aws-ofi-nccl" | head -1 || echo "   Could not determine version"
echo ""
echo "========================================="
echo "END DIAGNOSTIC"
echo "========================================="
'

# Run training
# NOTE: --export=ALL passes environment variables into the container
# CRITICAL: Must set LD_LIBRARY_PATH AND NCCL_NET inside container
srun --export=ALL \
        --container-image='/fsx/containers/nemo-container.sqsh' \
        --container-mounts='/fsx:/fsx' \
        --gres=gpu:8 \
        bash -c 'export LD_LIBRARY_PATH=/opt/amazon/aws-ofi-nccl/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH && export NCCL_NET="AWS Libfabric" && export TORCH_DISTRIBUTED_TIMEOUT=1800 && python3 /fsx/training/train_moe.py'
