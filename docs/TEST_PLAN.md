# Test Execution Plan

This document provides step-by-step instructions for running the deadlock reproduction test on both Kubernetes (EKS) and SLURM environments.

## Prerequisites

### Hardware Requirements

- **Minimum:** 2 nodes with NVIDIA GPUs (any GPU with NCCL support)
- **Recommended:** 2 x p5.48xlarge (8 H100 GPUs each) or equivalent
- AWS EFA networking for RDMA transport

### Software Requirements

- Docker or containerd
- Kubernetes with Kubeflow Training Operator, OR SLURM with Pyxis
- AWS CLI (for ECR access)
- kubectl (for Kubernetes)

## Phase 1: Build and Push Container

### 1.1 Build the Container

```bash
cd aws-ofi-nccl-deadlock-repro

# Build with NGC NeMo base image
docker build \
    --build-arg BASE_IMAGE=nvcr.io/nvidia/nemo:25.09.00 \
    -t aws-ofi-nccl-deadlock-test:latest \
    -f Dockerfile .

# Verify build
docker run --rm aws-ofi-nccl-deadlock-test:latest ls -la /opt/aws-ofi-nccl-1.14.0-vulnerable/lib/
docker run --rm aws-ofi-nccl-deadlock-test:latest ls -la /opt/amazon/aws-ofi-nccl/lib/
```

### 1.2 Push to Registry

**For ECR:**
```bash
# Set variables
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-2
REPO=aws-ofi-nccl-deadlock-test

# Create repository (if needed)
aws ecr create-repository --repository-name $REPO --region $REGION

# Login and push
aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

docker tag aws-ofi-nccl-deadlock-test:latest $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:latest
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:latest
```

**For SLURM (squashfs):**
```bash
# Convert to squashfs for Pyxis
enroot import -o aws-ofi-nccl-deadlock-test.sqsh dockerd://aws-ofi-nccl-deadlock-test:latest

# Or using docker save
docker save aws-ofi-nccl-deadlock-test:latest | gzip > aws-ofi-nccl-deadlock-test.tar.gz
```

## Phase 2: Run on Kubernetes (EKS)

### 2.1 Verify Cluster

```bash
# Check nodes
kubectl get nodes -l node.kubernetes.io/instance-type=p5.48xlarge

# Check GPU availability
kubectl describe node <node-name> | grep -A 10 "Allocated resources"

# Check EFA device plugin
kubectl get pods -n kube-system | grep efa
```

### 2.2 Update Manifests

Edit `k8s/pytorchjob.yaml`:
- Replace `<ACCOUNT>` with your AWS account ID
- Replace `<REGION>` with your region
- Update `nodeSelector` if needed for your cluster

### 2.3 Run Vulnerable Test

```bash
# Apply vulnerable test only
kubectl apply -f k8s/pytorchjob.yaml -n kubeflow

# Monitor pod creation
kubectl get pods -n kubeflow -w -l training.kubeflow.org/job-name=deadlock-test-vulnerable

# Follow logs
kubectl logs -f -n kubeflow -l training.kubeflow.org/job-name=deadlock-test-vulnerable,training.kubeflow.org/replica-type=master
```

### 2.4 Monitor for Deadlock

```bash
# Check GPU utilization (should drop to 0% if deadlocked)
kubectl exec -it <pod-name> -n kubeflow -- nvidia-smi

# Check iteration progress
kubectl logs -n kubeflow <pod-name> --tail=20

# If stuck for >5 minutes with 0% GPU: deadlock confirmed
```

### 2.5 Run Control Test

```bash
# Delete vulnerable test
kubectl delete pytorchjob deadlock-test-vulnerable -n kubeflow

# Run control (delete the vulnerable job definition from YAML first, or apply separately)
kubectl apply -f k8s/pytorchjob.yaml -n kubeflow

# Monitor - should complete without hanging
kubectl logs -f -n kubeflow -l training.kubeflow.org/job-name=deadlock-test-control
```

### 2.6 Cleanup

```bash
kubectl delete pytorchjob deadlock-test-vulnerable deadlock-test-control -n kubeflow
```

## Phase 3: Run on SLURM

### 3.1 Setup

```bash
# Copy container to shared filesystem
cp aws-ofi-nccl-deadlock-test.sqsh /shared/containers/

# Update run_test.sh with container path
sed -i 's|/path/to/|/shared/containers/|g' slurm/run_test.sh
```

### 3.2 Run Vulnerable Test

```bash
# Submit vulnerable test
LIBRARY_MODE=vulnerable sbatch slurm/run_test.sh

# Monitor
tail -f deadlock-test-*.out

# Check job status
squeue -u $USER
```

### 3.3 Run Control Test

```bash
# Submit control test
LIBRARY_MODE=fixed sbatch slurm/run_test.sh

# Monitor - should complete
tail -f deadlock-test-*.out
```

## Phase 4: Analyze Results

### Success Criteria

| Test | Expected Outcome | Indicates |
|------|------------------|-----------|
| Vulnerable + Injection | **DEADLOCK** (hang, 0% GPU) | Bug #1 reproduced |
| Control + Injection | **COMPLETES** | Fix verified |

### If Both Outcomes Occur

The hypothesis is confirmed:
- Version 1.14.0 has the deadlock bug
- Version 1.17.2 fixed the deadlock bug
- The upgrade was necessary and valuable

### If Vulnerable Test Doesn't Deadlock

Possible causes:
1. Error injection rate too low → Increase `MR_ERROR_RATE` to 10
2. Wrong library loaded → Check `LD_LIBRARY_PATH` and `ldd` output
3. Not enough memory pressure → Increase `ITERATIONS` and `HIDDEN_DIM`

```bash
# Debug: Check which library is loaded
kubectl exec <pod> -- bash -c 'ldd /opt/tests/moe_stress_test.py 2>&1 | grep nccl || echo "Check LD_LIBRARY_PATH"'
kubectl exec <pod> -- bash -c 'echo $LD_LIBRARY_PATH'

# Debug: Verify injection is working
kubectl logs <pod> 2>&1 | grep "MR_INJECT"
```

## Troubleshooting

### Pod stuck in Pending

```bash
# Check events
kubectl describe pod <pod-name> -n kubeflow

# Common issues:
# - Insufficient GPU/EFA resources (clear other workloads)
# - Image pull errors (check ECR auth)
# - Node selector mismatch (verify labels)
```

### NCCL Initialization Fails

```bash
# "Bootstrap: no socket interface found"
# - Verify hostNetwork: true in pod spec
# - Don't set NCCL_SOCKET_IFNAME (let auto-detect)

# "Unable to register memory"
# - Check EFA device mounts (/dev/infiniband, /dev/gdrdrv)
# - Verify privileged mode
```

### Both Tests Behave Identically

```bash
# Verify different libraries are actually loaded
# For vulnerable:
kubectl exec <vulnerable-pod> -- ls -la /opt/aws-ofi-nccl-1.14.0-vulnerable/lib/

# Check LD_LIBRARY_PATH at runtime
kubectl exec <pod> -- bash -c 'echo $LD_LIBRARY_PATH | cut -d: -f1'
```

## Appendix: Quick Reference

### Kubernetes Commands

```bash
# Watch pods
kubectl get pods -n kubeflow -w -l app=deadlock-test

# Get logs from all pods
kubectl logs -n kubeflow -l training.kubeflow.org/job-name=deadlock-test-vulnerable --all-containers

# Exec into pod
kubectl exec -it <pod> -n kubeflow -- bash

# Delete all test jobs
kubectl delete pytorchjobs -n kubeflow -l app=deadlock-test
```

### SLURM Commands

```bash
# Check queue
squeue -u $USER

# Cancel job
scancel <job-id>

# View output in real-time
tail -f deadlock-test-*.out

# Check node status
sinfo -N
```
