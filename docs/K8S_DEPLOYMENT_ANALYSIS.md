# Kubernetes Deployment Analysis

Pre-deployment analysis and best practice review for the aws-ofi-nccl deadlock reproduction test on EKS.

**Last Updated**: 2026-01-16
**Verified Against**: AWS EFA Cheatsheet, AWS EKS Best Practices, Kubeflow Documentation

## Executive Summary

| Category | Status | Critical Issues |
|----------|--------|-----------------|
| PyTorchJob Structure | ⚠️ Review | Missing elasticPolicy, no backoff limits |
| Resource Configuration | ⚠️ Review | EFA count may vary by instance, hugepages sizing |
| Network Configuration | ✅ OK | hostNetwork + ClusterFirstWithHostNet correct |
| Security Context | ✅ OK | Privileged + IPC_LOCK appropriate for RDMA |
| Anti-Affinity | ✅ OK | Pod anti-affinity ensures node spread |
| Environment Variables | ⚠️ Review | Missing critical NCCL settings |
| Device Mounts | ⚠️ Review | /dev/gdrdrv may not exist on all nodes |

---

## 1. PyTorchJob Configuration Issues

### 1.1 Missing Timeout Configuration

**Issue**: No `activeDeadlineSeconds` or TTL configuration.

```yaml
# CURRENT: No timeout
spec:
  nprocPerNode: "8"
  pytorchReplicaSpecs:
    ...

# RECOMMENDED: Add timeout for deadlock test
spec:
  nprocPerNode: "8"
  runPolicy:
    activeDeadlineSeconds: 1800  # 30 min - should deadlock before this
    ttlSecondsAfterFinished: 3600  # Cleanup after 1 hour
    backoffLimit: 0  # Don't retry on failure
```

**Why it matters**: If the deadlock test succeeds in triggering a hang, the job will run forever without a timeout. The `activeDeadlineSeconds` ensures Kubernetes kills the job after the specified time.

### 1.2 Missing Failure Policy

**Issue**: `restartPolicy: Never` is correct, but no `backoffLimit`.

```yaml
# RECOMMENDED: Add to runPolicy
runPolicy:
  backoffLimit: 0  # Explicit: don't retry
```

### 1.3 nprocPerNode Type

**Issue**: `nprocPerNode` should be a string, which it is. However, this must match the GPU count.

```yaml
# CURRENT (correct)
nprocPerNode: "8"  # String type required by Kubeflow

# VERIFY: Must match resource limits
resources:
  limits:
    nvidia.com/gpu: 8  # Must match nprocPerNode
```

---

## 2. Resource Configuration Issues

### 2.1 EFA Device Count

**Issue**: P5.48xlarge has 32 EFA devices, but other instance types differ.

| Instance Type | EFA Devices | GPUs |
|---------------|-------------|------|
| p5.48xlarge | 32 | 8 H100 |
| p4d.24xlarge | 4 | 8 A100 |
| p4de.24xlarge | 4 | 8 A100 |
| trn1.32xlarge | 8 | 16 Trainium |

```yaml
# CURRENT: Hardcoded for P5
resources:
  limits:
    vpc.amazonaws.com/efa: 32  # Only valid for p5.48xlarge!

# RECOMMENDATION: Verify instance type or parameterize
# For P4d testing, this would need to be 4
```

### 2.2 Hugepages Configuration

**Issue**: 5120Mi (5GB) hugepages may be insufficient for large-scale MR registration.

```yaml
# CURRENT
hugepages-2Mi: 5120Mi  # 2560 hugepages

# RECOMMENDED for stress testing (more MR headroom)
hugepages-2Mi: 10240Mi  # 5120 hugepages
```

**Verification needed**: Check node hugepages allocation:
```bash
kubectl get nodes -o json | jq '.items[].status.allocatable["hugepages-2Mi"]'
```

### 2.3 Memory Request vs Limit Mismatch

**Issue**: Memory has only `request`, no `limit`. This allows unbounded memory usage.

```yaml
# CURRENT
resources:
  limits:
    nvidia.com/gpu: 8
    # No memory limit!
  requests:
    memory: 200Gi

# RECOMMENDED: Add memory limit (or explicitly leave unlimited if intentional)
resources:
  limits:
    nvidia.com/gpu: 8
    memory: 400Gi  # Limit to prevent OOM-killing other pods
  requests:
    memory: 200Gi
```

### 2.4 CPU as String

**Issue**: CPU is specified as string `"48"` instead of integer.

```yaml
# CURRENT
cpu: "48"

# RECOMMENDED (both work, but integer is cleaner)
cpu: 48
```

---

## 3. Network Configuration

### 3.1 hostNetwork (Correct)

```yaml
hostNetwork: true
dnsPolicy: ClusterFirstWithHostNet
```

**Analysis**: This is CORRECT for EFA. AWS EFA requires direct host network access for RDMA traffic. The `ClusterFirstWithHostNet` ensures DNS still works with hostNetwork enabled.

### 3.2 Missing NCCL Network Settings

**Issue**: Several critical NCCL environment variables are missing.

```yaml
# CURRENT: Missing critical settings
env:
- name: NCCL_DEBUG
  value: "WARN"
- name: NCCL_TIMEOUT
  value: "300"

# RECOMMENDED: Add these for EFA
env:
- name: NCCL_DEBUG
  value: "WARN"
- name: NCCL_DEBUG_SUBSYS
  value: "INIT,NET"  # Useful for diagnosing EFA issues
- name: NCCL_TIMEOUT
  value: "300"
- name: NCCL_NET
  value: "AWS Libfabric"  # CRITICAL: Explicitly request aws-ofi-nccl
- name: FI_PROVIDER
  value: "efa"
- name: FI_EFA_USE_HUGE_PAGE
  value: "0"  # Prevents fork() memory issues
- name: NCCL_NVLS_ENABLE
  value: "0"  # Prevents NVLink peer access errors
- name: PYTORCH_CUDA_ALLOC_CONF
  value: "expandable_segments:True"
```

**Why NCCL_NET is critical**: Without explicitly setting `NCCL_NET="AWS Libfabric"`, NCCL may not load the aws-ofi-nccl plugin even if it's in `LD_LIBRARY_PATH`.

---

## 4. Device Mount Issues

### 4.1 /dev/gdrdrv May Not Exist

**Issue**: GPUDirect RDMA driver device `/dev/gdrdrv` may not exist on all EKS nodes.

```yaml
# CURRENT
- name: gdrdrv
  hostPath:
    path: /dev/gdrdrv

# PROBLEM: If /dev/gdrdrv doesn't exist, pod will fail to start

# RECOMMENDED: Make it optional or check node capability
volumes:
- name: gdrdrv
  hostPath:
    path: /dev/gdrdrv
    type: DirectoryOrCreate  # Won't fail if missing, but may create empty dir
```

**Better solution**: Use a node selector to ensure nodes have the nvidia-peermem driver:
```yaml
nodeSelector:
  node.kubernetes.io/instance-type: p5.48xlarge
  nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3  # Ensures H100 nodes
```

### 4.2 /dev/infiniband Path

**Issue**: On EFA nodes, the device path is `/dev/infiniband/` but devices are named `uverbsN` and `rdma_cmN`.

```bash
# Expected contents of /dev/infiniband on P5
/dev/infiniband/
├── rdma_cm
├── uverbs0
├── uverbs1
├── ...
└── uverbs31  # 32 devices on P5.48xlarge
```

**Verification**:
```bash
kubectl debug node/<node-name> -it --image=busybox -- ls -la /dev/infiniband/
```

---

## 5. Security Context Analysis

### 5.1 Current Settings (Appropriate)

```yaml
securityContext:
  privileged: true
  capabilities:
    add:
    - IPC_LOCK     # Required for RDMA memory registration
    - SYS_RESOURCE # Required for increasing locked memory limits
```

**Analysis**: These settings are necessary for RDMA/EFA workloads:
- `privileged: true` - Required for direct device access
- `IPC_LOCK` - Required for `ibv_reg_mr()` (memory registration)
- `SYS_RESOURCE` - Required for `ulimit -l unlimited`

### 5.2 Missing Settings (Consider Adding)

```yaml
securityContext:
  privileged: true
  capabilities:
    add:
    - IPC_LOCK
    - SYS_RESOURCE
    - NET_ADMIN      # May be needed for some EFA operations
    - SYS_PTRACE     # Useful for debugging hangs with gdb
```

---

## 6. Anti-Affinity Analysis

### 6.1 Current Configuration (Correct)

```yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchLabels:
          app: deadlock-test
      topologyKey: kubernetes.io/hostname
```

**Analysis**: This ensures master and worker pods are scheduled on different physical nodes, which is required for testing inter-node communication.

### 6.2 Potential Issue: Control vs Vulnerable Overlap

**Issue**: If both `deadlock-test-vulnerable` and `deadlock-test-control` are deployed simultaneously, they may compete for the same nodes.

```yaml
# CURRENT: Different app labels
# vulnerable: app: deadlock-test
# control:    app: deadlock-test-control

# RECOMMENDATION: Run them sequentially, not in parallel
# Or add explicit node exclusion between tests
```

---

## 7. Entrypoint Script Issues

### 7.1 Missing TORCH_DISTRIBUTED_TIMEOUT

The entrypoint.sh doesn't set `TORCH_DISTRIBUTED_TIMEOUT`, which could cause premature timeout before the deadlock test completes.

```bash
# ADD to entrypoint.sh
export TORCH_DISTRIBUTED_TIMEOUT=${TORCH_DISTRIBUTED_TIMEOUT:-1800}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

### 7.2 torchrun vs PyTorchJob Conflict

**Issue**: PyTorchJob automatically sets up distributed environment, but entrypoint also calls `torchrun`.

```bash
# CURRENT (may cause double-wrapping)
if [ "$GPUS_PER_NODE" -gt 1 ] && [ -n "$MASTER_ADDR" ]; then
    exec torchrun \
        --nnodes=$NNODES \
        ...
```

**PyTorchJob behavior**: Kubeflow Training Operator already handles `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, etc. Using `torchrun` again may cause conflicts.

**RECOMMENDATION**: Let PyTorchJob handle distributed setup:
```bash
# For PyTorchJob environments, just run Python directly
if [ -n "$PET_NNODES" ] || [ -n "$WORLD_SIZE" ]; then
    # PyTorchJob already set up distributed env
    exec python3 /opt/tests/$TEST_SCRIPT "$@"
else
    # Manual torchrun for non-Kubeflow environments
    exec torchrun ...
fi
```

---

## 8. Image and Registry Issues

### 8.1 Placeholder Image Path

```yaml
# CURRENT (will fail)
image: <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/aws-ofi-nccl-deadlock-test:latest

# MUST BE UPDATED before deployment
image: 058264135704.dkr.ecr.us-east-2.amazonaws.com/deadlock-test:latest
```

### 8.2 Image Pull Policy

```yaml
imagePullPolicy: Always
```

**Analysis**: `Always` is correct for testing (ensures latest image), but consider `IfNotPresent` for production to avoid registry issues.

### 8.3 Missing ImagePullSecrets

If ECR requires authentication:
```yaml
spec:
  template:
    spec:
      imagePullSecrets:
      - name: ecr-secret  # Must be created in namespace
```

---

## 9. Recommended YAML Changes

### Complete Updated YAML Section

```yaml
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: deadlock-test-vulnerable
  namespace: kubeflow
  labels:
    test-type: vulnerable
spec:
  nprocPerNode: "8"
  runPolicy:
    activeDeadlineSeconds: 1800    # 30 min timeout
    ttlSecondsAfterFinished: 3600  # Cleanup after 1 hour
    backoffLimit: 0                # No retries
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: Never
      template:
        spec:
          hostNetwork: true
          dnsPolicy: ClusterFirstWithHostNet
          nodeSelector:
            node.kubernetes.io/instance-type: p5.48xlarge

          containers:
          - name: pytorch
            image: 058264135704.dkr.ecr.us-east-2.amazonaws.com/deadlock-test:latest

            env:
            # Test configuration
            - name: LIBRARY_MODE
              value: "vulnerable"
            - name: MR_ERROR_RATE
              value: "50"
            - name: MR_ERROR_START
              value: "500"
            - name: TEST_SCRIPT
              value: "moe_stress_test.py"
            - name: GPUS_PER_NODE
              value: "8"
            - name: NNODES
              value: "2"
            - name: ITERATIONS
              value: "1000"

            # NCCL/EFA settings (CRITICAL)
            - name: NCCL_NET
              value: "AWS Libfabric"
            - name: NCCL_DEBUG
              value: "INFO"  # Use INFO for first run to verify EFA
            - name: NCCL_DEBUG_SUBSYS
              value: "INIT,NET"
            - name: NCCL_TIMEOUT
              value: "300"
            - name: NCCL_NVLS_ENABLE
              value: "0"
            - name: FI_PROVIDER
              value: "efa"
            - name: FI_EFA_USE_HUGE_PAGE
              value: "0"

            # PyTorch settings
            - name: TORCH_DISTRIBUTED_TIMEOUT
              value: "300"
            - name: TORCH_NCCL_ASYNC_ERROR_HANDLING
              value: "1"
            - name: PYTORCH_CUDA_ALLOC_CONF
              value: "expandable_segments:True"

            resources:
              limits:
                nvidia.com/gpu: 8
                vpc.amazonaws.com/efa: 32
                hugepages-2Mi: 5120Mi
                memory: 400Gi
              requests:
                nvidia.com/gpu: 8
                vpc.amazonaws.com/efa: 32
                memory: 200Gi
                cpu: 48
                hugepages-2Mi: 5120Mi
```

---

## 10. Pre-Deployment Checklist

### Before Applying YAML

- [ ] Update image path from placeholder to actual ECR URL
- [ ] Verify EKS cluster has Kubeflow Training Operator installed
- [ ] Verify nodes have the correct instance type (p5.48xlarge)
- [ ] Verify nodes have EFA devices available
- [ ] Check hugepages allocation on nodes
- [ ] Create ECR pull secret if needed
- [ ] Verify namespace exists (`kubectl get ns kubeflow`)

### Verification Commands

```bash
# Check Training Operator
kubectl get pods -n kubeflow -l control-plane=kubeflow-training-operator

# Check nodes and resources
kubectl get nodes -l node.kubernetes.io/instance-type=p5.48xlarge
kubectl describe node <node-name> | grep -A 20 "Allocatable:"

# Check EFA device plugin
kubectl get pods -n kube-system -l name=aws-efa-k8s-device-plugin

# Check hugepages
kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, hugepages: .status.allocatable["hugepages-2Mi"]}'
```

### Post-Deployment Monitoring

```bash
# Watch job status
kubectl get pytorchjob -n kubeflow -w

# Stream logs from master
kubectl logs -f -n kubeflow -l training.kubeflow.org/job-name=deadlock-test-vulnerable,training.kubeflow.org/replica-type=master

# Check for EFA initialization
kubectl logs -n kubeflow <pod-name> | grep -E "EFA|aws-ofi-nccl|fi_info"
```

---

## 11. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Job runs forever (deadlock works) | High | Medium | Add activeDeadlineSeconds |
| EFA not detected | Medium | High | Set NCCL_NET explicitly, use NCCL_DEBUG=INFO |
| Wrong library loaded | Medium | High | Verify with NCCL_DEBUG output |
| Hugepages exhausted | Low | High | Increase hugepages request |
| /dev/gdrdrv missing | Medium | Low | Already handled by privileged mode |
| Image pull failure | Low | High | Pre-pull image or create pull secret |

---

## 12. Quick Reference: Environment Variables

| Variable | Required | Value | Purpose |
|----------|----------|-------|---------|
| `NCCL_NET` | **YES** | `"AWS Libfabric"` | Force aws-ofi-nccl plugin |
| `FI_PROVIDER` | YES | `efa` | Select EFA provider |
| `FI_EFA_USE_HUGE_PAGE` | YES | `0` | Prevent fork() issues |
| `NCCL_TIMEOUT` | YES | `300` | Match test duration |
| `NCCL_NVLS_ENABLE` | YES | `0` | Prevent NVLink errors |
| `NCCL_DEBUG` | No | `INFO` | Debug EFA init |
| `TORCH_DISTRIBUTED_TIMEOUT` | YES | `300` | Match NCCL timeout |
| `LIBRARY_MODE` | YES | `vulnerable`/`fixed` | Select test version |
| `MR_ERROR_RATE` | YES | `50` | Inject error every N calls |

---

## 13. Internet Research Findings (2026-01-16)

This section documents findings from official AWS documentation and community best practices.

### 13.1 AWS EFA Cheatsheet Alignment

**Source**: [AWS EFA Cheatsheet](https://github.com/aws-samples/awsome-distributed-training/blob/main/1.architectures/efa-cheatsheet.md)

| Our Setting | AWS Recommendation | Status |
|-------------|-------------------|--------|
| `FI_EFA_USE_HUGE_PAGE=0` | Set to 0 for fork()/GC issues | ✅ Aligned |
| `FI_PROVIDER=efa` | Only needed for aws-ofi-nccl ≤1.5.0 | ⚠️ Not harmful, but unnecessary for 1.17.2 |
| `NCCL_PROTO` | NOT SET | ✅ Correct (disables tuner if set) |
| `NCCL_ALGO` | NOT SET | ✅ Correct (disables tuner if set) |
| `NCCL_SOCKET_IFNAME` | NOT SET | ✅ Correct (P5 doesn't use eth0) |
| `FI_EFA_USE_DEVICE_RDMA` | NOT SET | ✅ Correct (auto-enabled in libfabric ≥1.18.0) |

**Critical Warning from AWS**:
> "Do NOT set `RDMAV_FORK_SAFE=1` - can break things on newer kernels"

### 13.2 NCCL_NVLS_ENABLE Justification

**Source**: [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

Setting `NCCL_NVLS_ENABLE=0` is appropriate because:
1. NVLS (NVLink SHARP) can cause hangs in some configurations
2. Known issue: "NCCL 2.19.4 hang if NCCL_NVLS_ENABLE=1" ([GitHub #1197](https://github.com/NVIDIA/nccl/issues/1197))
3. AWS P5 instances may have fabric manager issues with NVLS enabled
4. Disabling NVLS still allows other NVLink transports to function

### 13.3 HyperPod Training Operator Considerations

**Source**: [AWS HyperPod Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks.html)

If deploying on HyperPod EKS:
- Consider using `HyperPodPyTorchJob` instead of standard `PyTorchJob`
- HyperPod provides automatic EFA health checks
- Job auto-resume on EFA failures is available

**Note**: Our current YAML uses standard Kubeflow `PyTorchJob`, which works but doesn't have HyperPod's auto-recovery features.

### 13.4 EKS Best Practices Alignment

**Source**: [AWS EKS Best Practices - AI/ML](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html)

| Best Practice | Our Status |
|--------------|------------|
| Use cluster placement groups | ⚠️ Not configured (needs node-level setup) |
| Implement checkpointing | N/A (stress test, not training) |
| Extended termination grace period | ✅ Added `terminationGracePeriodSeconds: 30` |
| Monitor RDMA errors | ⚠️ Requires CloudWatch/DCGM setup |
| Disable Istio sidecar injection | ⚠️ May need `sidecar.istio.io/inject: "false"` |

### 13.5 Version Compatibility Matrix

**Source**: [EFA Best Practices](https://swsmith.cc/posts/efa-best-practices.html)

For P5.48xlarge (our target):

| Component | Minimum | Recommended | Our Container |
|-----------|---------|-------------|---------------|
| CUDA | ≥12.0 | 12.2+ | ✅ 12.9 |
| NCCL | ≥2.18.0 | 2.18.5+ | ✅ 2.27.3 |
| aws-ofi-nccl | ≥1.7.2 | 1.7.3+ | ✅ 1.17.2 |
| libfabric | ≥1.18.0 | 1.22+ | ✅ 1.22.0 |

### 13.6 Variables We Should NOT Set

Based on AWS documentation, these variables should NOT be set:

| Variable | Reason |
|----------|--------|
| `RDMAV_FORK_SAFE=1` | Breaks on newer kernels |
| `NCCL_PROTO=simple` | Disables aws-ofi-nccl tuner |
| `NCCL_ALGO=*` | Disables aws-ofi-nccl tuner |
| `NCCL_SOCKET_IFNAME=eth0` | P5 instances don't use eth0 |
| `FI_EFA_USE_DEVICE_RDMA=1` | Auto-enabled in libfabric ≥1.18.0 |
| `NCCL_P2P_DISABLE=1` | Disables NVLink (massive perf loss) |
| `NCCL_IB_DISABLE=1` | Not applicable to EFA |

### 13.7 Potential Issues Identified

#### Issue 1: Istio Sidecar Injection
**Risk**: PyTorchJob may fail if Istio auto-injects sidecars
**Mitigation**: Add annotation if Istio is installed:
```yaml
annotations:
  sidecar.istio.io/inject: "false"
```

#### Issue 2: Placement Groups
**Risk**: Without placement groups, inter-node latency may be higher
**Mitigation**: Ensure EKS nodes are in cluster placement groups (node-level config)

#### Issue 3: EFA Device Plugin
**Risk**: EFA resources won't be available without device plugin
**Verification**:
```bash
kubectl get pods -n kube-system -l name=aws-efa-k8s-device-plugin
```

### 13.8 Recommended Pre-Flight Checks

Based on internet research, run these before deployment:

```bash
# 1. Verify EFA device plugin is running
kubectl get pods -n kube-system | grep efa

# 2. Check EFA resources on nodes
kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, efa: .status.allocatable["vpc.amazonaws.com/efa"]}'

# 3. Verify Training Operator
kubectl get crd pytorchjobs.kubeflow.org

# 4. Check for Istio (if present, need annotation)
kubectl get ns istio-system 2>/dev/null && echo "Istio detected - add sidecar.istio.io/inject: false"

# 5. Verify hugepages
kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, hugepages: .status.allocatable["hugepages-2Mi"]}'
```

---

## 14. References

- [AWS EFA Cheatsheet](https://github.com/aws-samples/awsome-distributed-training/blob/main/1.architectures/efa-cheatsheet.md)
- [AWS EKS Best Practices - AI/ML](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html)
- [aws-ofi-nccl GitHub](https://github.com/aws/aws-ofi-nccl)
- [Kubeflow PyTorchJob](https://www.kubeflow.org/docs/components/trainer/legacy-v1/user-guides/pytorch/)
- [NVIDIA NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [SageMaker HyperPod EKS](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks.html)
- [EFA Best Practices (Sean Smith)](https://swsmith.cc/posts/efa-best-practices.html)
