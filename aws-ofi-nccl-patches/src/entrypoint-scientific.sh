#!/bin/bash
# entrypoint-scientific.sh - Entrypoint for scientific deadlock testing
#
# Environment variables:
#   FI_MR_INJECT_ENABLE=1    - Enable fault injection
#   FI_MR_INJECT_START=500   - Start injecting after N calls
#   FI_MR_INJECT_RATE=50     - Inject every N calls after start
#   TEST_SCRIPT              - Test script to run (default: moe_stress_test.py)

set -e

# Read version
VERSION=$(cat /opt/aws-ofi-nccl-version.txt 2>/dev/null || echo "unknown")

echo "============================================================"
echo "  Scientific Deadlock Test - aws-ofi-nccl ${VERSION}"
echo "============================================================"

# Setup library paths
export LD_LIBRARY_PATH=/opt/amazon/aws-ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}

# Check if fault injection is enabled
if [ "${FI_MR_INJECT_ENABLE:-0}" = "1" ]; then
    echo ""
    echo ">>> FAULT INJECTION ENABLED <<<"
    echo "    LD_PRELOAD: libfi_mr_inject.so"
    echo "    Start after: ${FI_MR_INJECT_START:-500} calls"
    echo "    Inject every: ${FI_MR_INJECT_RATE:-50} calls"
    echo ""
    export LD_PRELOAD=/opt/amazon/aws-ofi-nccl/lib/libfi_mr_inject.so
else
    echo ""
    echo ">>> Fault injection DISABLED <<<"
    echo "    Set FI_MR_INJECT_ENABLE=1 to enable"
    echo ""
fi

# Show configuration
echo "Configuration:"
echo "  aws-ofi-nccl version: ${VERSION}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "  LD_PRELOAD: ${LD_PRELOAD:-<none>}"
echo "  NCCL_NET: ${NCCL_NET:-<default>}"
echo ""

# Verify library version
echo "Library verification:"
strings /opt/amazon/aws-ofi-nccl/lib/libnccl-net.so | grep "aws-ofi-nccl" | head -1
echo ""
echo "============================================================"
echo ""

# Get test script
TEST_SCRIPT="${TEST_SCRIPT:-moe_stress_test.py}"
ITERATIONS="${ITERATIONS:-5000}"
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# Determine node rank
if [ -n "${GROUP_RANK}" ]; then
    NODE_RANK=${GROUP_RANK}
elif [ -n "${RANK}" ]; then
    NODE_RANK=$((RANK / GPUS_PER_NODE))
else
    NODE_RANK=0
fi

# Determine master address
if [ -n "${MASTER_ADDR}" ]; then
    MASTER="${MASTER_ADDR}"
elif [ -n "${MY_POD_IP}" ] && [ "${NODE_RANK}" = "0" ]; then
    MASTER="${MY_POD_IP}"
else
    # Try to get from hostname
    MASTER=$(hostname -i 2>/dev/null || echo "localhost")
fi

MASTER_PORT="${MASTER_PORT:-23456}"

echo "Running test: ${TEST_SCRIPT}"
echo ""
echo "Launching ${GPUS_PER_NODE} processes via torchrun..."
echo "  --nnodes=${NNODES}"
echo "  --nproc_per_node=${GPUS_PER_NODE}"
echo "  --node_rank=${NODE_RANK}"
echo "  --master_addr=${MASTER}"
echo "  --master_port=${MASTER_PORT}"
echo ""

# Export for test script
export ITERATIONS
export NNODES
export GPUS_PER_NODE

# Run with torchrun
exec torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER} \
    --master_port=${MASTER_PORT} \
    /workspace/${TEST_SCRIPT}
