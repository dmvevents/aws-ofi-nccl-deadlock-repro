# AWS OFI NCCL Deadlock Reproduction Test

This repository provides tools to reproduce the memory registration deadlock bug in aws-ofi-nccl v1.14.0, which was fixed in v1.17.2 via [PR #968](https://github.com/aws/aws-ofi-nccl/pull/968).

## Background

### The Bug

In aws-ofi-nccl versions prior to 1.17.2, a deadlock could occur in the RDMA transport layer when memory registration (`fi_mr_reg`) failed under high load. The issue was in `nccl_ofi_rdma.cpp`:

```cpp
// VULNERABLE CODE (v1.14.0)
static int reg_mr(nccl_net_ofi_rdma_ep_t *ep, void *data, size_t size, int type,
                  nccl_net_ofi_rdma_mr_handle_t **mhandle) {
    pthread_mutex_lock(&ep->lock);  // Lock acquired

    ret = fi_mr_reg(...);
    if (ret != 0) {
        NCCL_OFI_WARN("fi_mr_reg failed");
        return ret;  // BUG: Lock not released!
    }

    pthread_mutex_unlock(&ep->lock);
    return 0;
}
```

When `fi_mr_reg` fails (due to resource exhaustion, memory pressure, or transient EFA errors), the mutex is never released, causing all subsequent NCCL operations on that endpoint to deadlock.

### The Fix (PR #968)

```cpp
// FIXED CODE (v1.17.2)
static int reg_mr(nccl_net_ofi_rdma_ep_t *ep, void *data, size_t size, int type,
                  nccl_net_ofi_rdma_mr_handle_t **mhandle) {
    pthread_mutex_lock(&ep->lock);

    ret = fi_mr_reg(...);
    if (ret != 0) {
        NCCL_OFI_WARN("fi_mr_reg failed");
        pthread_mutex_unlock(&ep->lock);  // FIX: Release lock on error
        return ret;
    }

    pthread_mutex_unlock(&ep->lock);
    return 0;
}
```

## Test Strategy

Since triggering natural `fi_mr_reg` failures requires specific conditions (memory pressure, EFA resource exhaustion), we use **fault injection** to simulate these failures:

1. Build a container with both vulnerable (1.14.0) and fixed (1.17.2) aws-ofi-nccl
2. Use LD_PRELOAD to intercept `fi_mr_reg` calls and inject failures
3. Run a workload that generates high MR registration churn (MoE All-to-All patterns)
4. Compare behavior: vulnerable version should deadlock, fixed version should recover

## Requirements

- 2+ nodes with NVIDIA GPUs (H100/A100 recommended)
- AWS EFA networking
- Kubernetes with Kubeflow Training Operator, OR SLURM with Pyxis
- Docker/containerd

## Quick Start

### Build the Container

```bash
# Using NVIDIA NGC NeMo as base (includes PyTorch, NCCL, EFA)
docker build -f Dockerfile \
    --build-arg BASE_IMAGE=nvcr.io/nvidia/nemo:25.09.00 \
    -t aws-ofi-nccl-deadlock-test:latest .
```

### Push to Registry

```bash
# ECR example
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-2

aws ecr create-repository --repository-name aws-ofi-nccl-deadlock-test --region $REGION
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker tag aws-ofi-nccl-deadlock-test:latest $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/aws-ofi-nccl-deadlock-test:latest
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/aws-ofi-nccl-deadlock-test:latest
```

### Run on Kubernetes (EKS)

```bash
# Update image path in YAML first
sed -i "s|<ACCOUNT>|$ACCOUNT|g; s|<REGION>|$REGION|g" k8s/pytorchjob.yaml

# Deploy vulnerable test
kubectl apply -f k8s/pytorchjob.yaml -n kubeflow

# Monitor
kubectl logs -f -n kubeflow -l training.kubeflow.org/job-name=deadlock-test-vulnerable

# Cleanup
kubectl delete pytorchjob deadlock-test-vulnerable deadlock-test-control -n kubeflow
```

### Run on SLURM

```bash
# Simple stress test
LIBRARY_MODE=vulnerable sbatch slurm/run_test.sh

# Full NeMo MoE training (requires NeMo container)
sbatch slurm/train_moe.sh
```

## Test Modes

### 1. Stress Test (Recommended for Quick Validation)

Uses a synthetic MoE-like workload with fault injection:

```bash
# Kubernetes
kubectl apply -f k8s/pytorchjob.yaml -n kubeflow

# SLURM
LIBRARY_MODE=vulnerable MR_ERROR_RATE=50 sbatch slurm/run_test.sh
```

### 2. Full MoE Training (Production Workload)

Uses the MoE17B 17B MoE model configuration with NeMo:

```bash
# SLURM with NeMo container
CONTAINER_IMAGE=/path/to/nemo.sqsh sbatch slurm/train_moe.sh
```

The MoE17B model is a 17B parameter MoE with:
- 21 layers (1 dense + 20 MoE)
- 64 experts with top-6 routing
- Expert parallelism across 8 GPUs

See `nemo/model_config.py` for full architecture details.

## Test Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `LIBRARY_MODE` | `vulnerable` (1.14.0) or `fixed` (1.17.2) | `fixed` |
| `MR_ERROR_RATE` | Inject error every N calls (0=disabled) | `0` |
| `MR_ERROR_START` | Start injection after N calls | `500` |
| `NCCL_TIMEOUT` | NCCL timeout in seconds | `300` |
| `NUM_NODES` | Number of nodes for training | `2` |
| `MAX_STEPS` | Training steps | `1000` |

## Expected Results

| Test | Configuration | Expected Outcome |
|------|---------------|------------------|
| Vulnerable | `LIBRARY_MODE=vulnerable MR_ERROR_RATE=50` | **DEADLOCK** within minutes |
| Control | `LIBRARY_MODE=fixed MR_ERROR_RATE=50` | **COMPLETES** (errors logged but recovers) |

## File Structure

```
.
├── Dockerfile                 # Multi-version container build
├── README.md                  # This file
├── LICENSE                    # Apache 2.0
├── docs/
│   ├── HYPOTHESIS.md          # Two-bug hypothesis analysis
│   ├── CODE_ANALYSIS.md       # Technical bug analysis
│   └── TEST_PLAN.md           # Detailed test execution guide
├── src/
│   ├── inject_mr_errors.c     # LD_PRELOAD fault injection library
│   ├── moe_stress_test.py     # MoE workload to trigger MR churn
│   └── entrypoint.sh          # Container entrypoint
├── nemo/
│   ├── train_moe.py           # MoE17B 17B MoE training script
│   └── model_config.py        # Model architecture configuration
├── k8s/
│   └── pytorchjob.yaml        # Kubernetes PyTorchJob manifests
└── slurm/
    ├── run_test.sh            # Simple stress test launcher
    └── train_moe.sh           # Full NeMo training launcher
```

## Documentation

- **[docs/HYPOTHESIS.md](docs/HYPOTHESIS.md)** - Analysis proving two distinct bugs existed
- **[docs/CODE_ANALYSIS.md](docs/CODE_ANALYSIS.md)** - Deep dive into the deadlock bug
- **[docs/TEST_PLAN.md](docs/TEST_PLAN.md)** - Step-by-step test execution guide

## Troubleshooting

### Test completes without deadlock (vulnerable mode)

1. Increase error rate: `MR_ERROR_RATE=10`
2. Decrease start delay: `MR_ERROR_START=100`
3. Verify injection is working: check logs for `[MR_INJECT]` messages
4. Verify correct library loaded:
   ```bash
   # Should show /opt/aws-ofi-nccl-1.14.0-vulnerable/lib/libnccl-net.so
   kubectl exec <pod> -- bash -c 'echo $LD_LIBRARY_PATH'
   ```

### Image pull failures (Kubernetes)

```bash
kubectl create secret docker-registry ecr-secret \
  --docker-server=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com \
  --docker-username=AWS \
  --docker-password=$(aws ecr get-login-password --region $REGION)
```

### EFA not working

```bash
# Check EFA devices
kubectl exec <pod> -- ls /dev/infiniband/
kubectl exec <pod> -- /opt/amazon/efa/bin/fi_info -p efa

# Check aws-ofi-nccl version
kubectl exec <pod> -- strings /opt/amazon/aws-ofi-nccl/lib/libnccl-net.so | grep aws-ofi-nccl
```

### Training hangs but no MR injection messages

The workload may not be hitting `fi_mr_reg`. Try:
- Increase `HIDDEN_DIM` or `NUM_EXPERTS` to increase memory allocation
- Run for more iterations
- Use the full NeMo training instead of the stress test

## References

- [aws-ofi-nccl PR #968](https://github.com/aws/aws-ofi-nccl/pull/968) - The fix
- [aws-ofi-nccl GitHub](https://github.com/aws/aws-ofi-nccl)
- [EFA Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

## License

Apache 2.0
