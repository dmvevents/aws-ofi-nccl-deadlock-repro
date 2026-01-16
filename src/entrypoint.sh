#!/bin/bash
#
# Entrypoint for aws-ofi-nccl deadlock reproduction test
#
# Handles library selection and distributed setup for both
# Kubernetes (PyTorchJob) and SLURM environments.

set -e

echo "============================================================"
echo "AWS OFI NCCL Deadlock Reproduction Test"
echo "============================================================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

# Load library paths
if [ -f /opt/library_paths.env ]; then
    . /opt/library_paths.env
else
    VULNERABLE_LIB=/opt/aws-ofi-nccl-1.14.0-vulnerable
    FIXED_LIB=/opt/amazon/aws-ofi-nccl
    EFA_LIB=/opt/amazon/efa
fi

echo "Library paths:"
echo "  VULNERABLE: $VULNERABLE_LIB"
echo "  FIXED:      $FIXED_LIB"
echo "  EFA:        $EFA_LIB"
echo ""

# ============================================================
# Distributed configuration
# ============================================================

# Try to detect distributed settings from environment
# Kubernetes (PyTorchJob)
if [ -n "$MASTER_ADDR" ]; then
    echo "Detected Kubernetes/PyTorchJob environment"
# SLURM
elif [ -n "$SLURM_JOB_ID" ]; then
    echo "Detected SLURM environment"
    # Get master from first node in allocation
    MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)}
    NODE_RANK=${SLURM_NODEID:-0}
    NNODES=${SLURM_NNODES:-1}
fi

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-${GROUP_RANK:-${SLURM_NODEID:-0}}}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Distributed config:"
echo "  NNODES:        $NNODES"
echo "  NODE_RANK:     $NODE_RANK"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  MASTER_ADDR:   ${MASTER_ADDR:-<auto>}"
echo "  MASTER_PORT:   $MASTER_PORT"
echo ""

# ============================================================
# Library selection (CRITICAL)
# ============================================================

LIBRARY_MODE=${LIBRARY_MODE:-fixed}
echo "Library mode: $LIBRARY_MODE"

if [ "$LIBRARY_MODE" = "vulnerable" ]; then
    echo ">>> Using VULNERABLE aws-ofi-nccl 1.14.0"
    export LD_LIBRARY_PATH=$VULNERABLE_LIB/lib:$EFA_LIB/lib:$LD_LIBRARY_PATH

    if [ ! -f "$VULNERABLE_LIB/lib/libnccl-net.so" ]; then
        echo "ERROR: Vulnerable library not found!"
        ls -la "$VULNERABLE_LIB/lib/" 2>/dev/null || echo "  Directory does not exist"
        exit 1
    fi
else
    echo ">>> Using FIXED aws-ofi-nccl 1.17.2"
    export LD_LIBRARY_PATH=$FIXED_LIB/lib:$EFA_LIB/lib:$LD_LIBRARY_PATH

    if [ ! -f "$FIXED_LIB/lib/libnccl-net.so" ]; then
        echo "ERROR: Fixed library not found!"
        exit 1
    fi
fi

# ============================================================
# Error injection setup
# ============================================================

MR_ERROR_RATE=${MR_ERROR_RATE:-0}
MR_ERROR_START=${MR_ERROR_START:-500}

if [ "$MR_ERROR_RATE" != "0" ] && [ -f /opt/inject_mr_errors.so ]; then
    echo ">>> Error injection ENABLED: 1/$MR_ERROR_RATE (start after $MR_ERROR_START)"
    export LD_PRELOAD=/opt/inject_mr_errors.so
    export MR_ERROR_RATE
    export MR_ERROR_START
else
    echo ">>> Error injection DISABLED"
fi

# ============================================================
# NCCL/EFA configuration
# ============================================================

export NCCL_NET="AWS Libfabric"
export FI_PROVIDER=efa
export FI_EFA_USE_HUGE_PAGE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-300}
export NCCL_NVLS_ENABLE=0

echo ""
echo "Environment:"
echo "  LD_LIBRARY_PATH: $(echo $LD_LIBRARY_PATH | cut -d: -f1-2)..."
echo "  LD_PRELOAD:      ${LD_PRELOAD:-<none>}"
echo "  NCCL_NET:        $NCCL_NET"
echo "  FI_PROVIDER:     $FI_PROVIDER"
echo "  NCCL_DEBUG:      $NCCL_DEBUG"
echo "  NCCL_TIMEOUT:    $NCCL_TIMEOUT"
echo ""

# Verify library that will load
echo "Library verification:"
FIRST_LIB_PATH=$(echo $LD_LIBRARY_PATH | cut -d: -f1)
if [ -f "$FIRST_LIB_PATH/libnccl-net.so" ]; then
    echo "  Will load: $FIRST_LIB_PATH/libnccl-net.so"
    ls -la "$FIRST_LIB_PATH/libnccl-net.so"
else
    echo "  WARNING: libnccl-net.so not found in $FIRST_LIB_PATH"
fi

# Check for ldconfig override (common NGC issue)
if ldconfig -p 2>/dev/null | grep -q libnccl-net.so; then
    echo "  WARNING: ldconfig has libnccl-net.so entry (may override LD_LIBRARY_PATH)"
    ldconfig -p | grep libnccl-net.so
fi

echo ""
echo "============================================================"
echo ""

# ============================================================
# Run the test
# ============================================================

TEST_SCRIPT=${TEST_SCRIPT:-moe_stress_test.py}
echo "Running: $TEST_SCRIPT"
echo ""

if [ "$GPUS_PER_NODE" -gt 1 ] && [ -n "$MASTER_ADDR" ]; then
    echo "Launching via torchrun..."
    exec torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        /opt/tests/$TEST_SCRIPT "$@"
elif [ -n "$SLURM_JOB_ID" ]; then
    echo "Launching via SLURM srun..."
    exec python3 /opt/tests/$TEST_SCRIPT "$@"
else
    echo "Launching single process..."
    exec python3 /opt/tests/$TEST_SCRIPT "$@"
fi
