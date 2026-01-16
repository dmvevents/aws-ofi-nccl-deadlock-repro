# Scientific Test Plan: aws-ofi-nccl MR Deadlock Bug (PR #968)

**Date:** January 16, 2026
**Purpose:** Scientifically prove that the MR deadlock bug would still cause problems if the code still existed

---

## Executive Summary

We will prove that the aws-ofi-nccl upgrade from 1.14.0 → 1.17.2 was necessary by:
1. Taking the fixed v1.17.2 codebase
2. **Reverting ONLY PR #968** (the deadlock fix)
3. Injecting MR registration errors via LD_PRELOAD
4. Demonstrating that the reverted code **deadlocks** while the fixed code **does not**

---

## The Bug (PR #968)

**File:** `src/nccl_ofi_rdma.cpp` (and `nccl_ofi_mr.cpp`)
**Commit:** `06e3ebd1f934bc8334f6fb9b670b573c6b77c6ae`
**Merged:** August 20, 2025

```cpp
// BEFORE FIX (v1.14.0) - DEADLOCK
int reg_mr_on_device(...) {
    pthread_mutex_lock(&mr_cache->lock);    // Lock acquired

    if (error_condition) {
        dereg_mr(mr_handle);                 // Tries to acquire SAME lock
        // DEADLOCK - infinite wait
    }

    pthread_mutex_unlock(&mr_cache->lock);
}

// AFTER FIX (v1.17.2) - NO DEADLOCK
int reg_mr_on_device(...) {
    pthread_mutex_lock(&mr_cache->lock);

    if (error_condition) {
        dereg_mr_no_lock(mr_handle);         // Uses no-lock variant
        // No deadlock - proceeds normally
    }

    pthread_mutex_unlock(&mr_cache->lock);
}
```

---

## Test Matrix

| Test | Code Version | PR #968 Status | Fault Injection | Expected Result |
|------|--------------|----------------|-----------------|-----------------|
| **A** | v1.17.2 | REVERTED | Enabled | **DEADLOCK** |
| **B** | v1.17.2 | Intact | Enabled | **No deadlock** |

**Scientific proof:** Same codebase, same fault injection, only difference is PR #968.

---

## Implementation

### Files Created

```
aws-ofi-nccl-patches/
├── Dockerfile.revert-test      # Build with PR #968 reverted
├── Dockerfile.scientific-test  # Build v1.14.0 directly
├── fi_mr_inject.c              # LD_PRELOAD fault injection library
├── entrypoint-scientific.sh    # Container entrypoint
├── moe_stress_test.py          # MoE workload stress test
├── rdma_stress_test.py         # RDMA-heavy stress test
└── pytorchjob-scientific-test.yaml  # K8s deployment
```

### Build Commands

```bash
cd /home/ubuntu/slurm-test/bharatgen/nccl-deadlock-experiment/aws-ofi-nccl-patches

# Test A: v1.17.2 with PR #968 REVERTED (should deadlock)
docker build \
  --build-arg REVERT_PR968=true \
  -t deadlock-test:revert-pr968 \
  -f Dockerfile.revert-test .

# Test B: v1.17.2 CONTROL (should NOT deadlock)
docker build \
  --build-arg REVERT_PR968=false \
  -t deadlock-test:control \
  -f Dockerfile.revert-test .
```

### Fault Injection Mechanism

The `fi_mr_inject.c` library intercepts `fi_mr_regattr()` calls:

```c
// Environment variables:
FI_MR_INJECT_ENABLE=1     // Enable injection
FI_MR_INJECT_START=500    // Start after 500 calls (let init complete)
FI_MR_INJECT_RATE=100     // Inject every 100 calls (1% error rate)

// What it does:
// 1. Intercepts fi_mr_regattr() via LD_PRELOAD
// 2. After START calls, returns -FI_ENOMEM every RATE calls
// 3. This triggers the error path in reg_mr_on_device()
// 4. If PR #968 is reverted: DEADLOCK
// 5. If PR #968 is intact: graceful error handling
```

### Deployment

```yaml
# Key environment variables in PyTorchJob
env:
- name: FI_MR_INJECT_ENABLE
  value: "1"
- name: FI_MR_INJECT_START
  value: "500"
- name: FI_MR_INJECT_RATE
  value: "100"
- name: NCCL_TIMEOUT
  value: "120"  # Short timeout to detect deadlock
```

---

## Expected Outcomes

### Test A (REVERTED PR #968)
- Job starts normally
- After ~500 MR registrations, fault injection begins
- Within minutes, all GPUs go idle
- Processes stuck on `pthread_mutex_lock`
- NCCL times out after 120 seconds
- **Conclusion: Bug is real and would cause production failures**

### Test B (CONTROL)
- Job starts normally
- Fault injection triggers MR errors
- Errors are handled gracefully (may see warnings)
- Training continues or fails gracefully
- No deadlock, no hung processes
- **Conclusion: PR #968 fix works**

---

## Verification Commands

When test is running, verify deadlock state:

```bash
# Check GPU utilization (should be 0% if deadlocked)
kubectl exec -it deadlock-scientific-master-0 -n kubeflow -- nvidia-smi

# Check for hung threads
kubectl exec -it deadlock-scientific-master-0 -n kubeflow -- \
  bash -c 'PID=$(pgrep -f python); cat /proc/$PID/stack'

# Expected in deadlock:
#   futex_wait_queue
#   pthread_mutex_lock
#   dereg_mr
#   reg_mr_on_device
```

---

## Relationship to Production Issue

| Production Observation | Test Correlation |
|------------------------|------------------|
| Random hangs at 30-2000 steps | MR errors are probabilistic |
| 64-node scale increased frequency | More GPUs = more MR operations |
| All GPUs idle during hang | Deadlock blocks all threads |
| Upgrade to 1.17.2 fixed it | PR #968 removed the deadlock |

---

## Next Steps After Testing

1. **If deadlock confirmed:** Document as evidence that upgrade was necessary
2. **Build v1.17.3 test:** Also test PR #1087 (schedule leak fix)
3. **Create report:** Summarize findings with timestamps and logs
4. **Share with team:** Provide evidence for the upgrade decision

---

## Quick Reference

```bash
# Refresh AWS credentials (if expired)
aws sso login --profile default

# Refresh kubectl
aws eks update-kubeconfig --name hyperpod-eks-cluster --region us-east-2

# Deploy test
kubectl apply -f pytorchjob-scientific-test.yaml

# Monitor
kubectl logs -f deadlock-scientific-master-0 -n kubeflow

# Check for deadlock indicators
kubectl exec deadlock-scientific-master-0 -n kubeflow -- \
  grep -c "INJECTING" /proc/1/fd/2  # Count injected errors
```

---

**Document Version:** 1.0
**Author:** Claude Code
