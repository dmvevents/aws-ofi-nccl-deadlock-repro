# Scientific Proof: aws-ofi-nccl Deadlock Bug (PR #968)

**Date:** 2026-01-16
**Status:** Implementation in Progress

---

## Objective

Scientifically prove that the aws-ofi-nccl upgrade from **1.14.0 → 1.17.2** was necessary by demonstrating that Bug #1 (MR deadlock, PR #968) causes real problems when MR registration errors occur.

---

## The Bug

### Root Cause

In aws-ofi-nccl versions < 1.17.2, a deadlock occurs in `reg_mr_on_device()` when memory registration fails:

```cpp
// File: src/nccl_ofi_rdma.cpp, line ~2615

// BUGGY CODE (v1.14.0):
int reg_mr_on_device(...) {
    // Lock is held by caller (reg_mr)
    // mr_cache->lock is ALREADY HELD

    ret = fi_mr_regattr(...);  // This fails with -FI_ENOMEM
    if (ret != 0) {
        goto error;
    }
    return 0;

error:
    (void) this->dereg_mr(ret_handle);  // ❌ TRIES TO ACQUIRE SAME LOCK
    return ret;                          // → DEADLOCK (infinite wait)
}
```

### The Fix (PR #968)

```cpp
// FIXED CODE (v1.17.2):
error:
    (void) dereg_mr_no_lock(ret_handle);  // ✅ Doesn't acquire lock
    return ret;                            // → Graceful cleanup
```

**The entire fix is ONE LINE** - changing `dereg_mr()` to `dereg_mr_no_lock()`.

---

## Scientific Testing Approach

### The Problem

The bug only triggers when `fi_mr_regattr()` returns an error. Under normal operation, MR registration rarely fails, so the bug path is never exercised.

### The Solution: Controlled Fault Injection

```
┌─────────────────────────────────────────────────────────────┐
│                    SCIENTIFIC TEST DESIGN                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Independent Variable: Bug presence (with-bug vs fixed)      │
│  Controlled Variable:  Fault injection (same rate for both)  │
│  Dependent Variable:   System behavior (deadlock vs success) │
│                                                              │
│  Test A: v1.17.2 WITH bug + fault injection → DEADLOCK       │
│  Test B: v1.17.2 WITHOUT bug + fault injection → SUCCESS     │
│                                                              │
│  Same faults + Different outcome = Bug proven                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Step 1: Patch AWS Libfabric for Fault Injection

We patch the EFA provider in AWS libfabric (`prov/efa/src/efa_mr.c`) to inject `-FI_ENOMEM` errors at a controlled rate.

**Header file: `efa_mr_fault_inject.h`**
```c
static int _fi_mr_should_inject(void) {
    if (!enabled) return 0;
    count = atomic_fetch_add(&call_count, 1);
    if (count < start) return 0;
    return ((count - start) % rate) == 0;
}
```

**Injection point in `efa_mr_regattr()`:**
```c
// After allocating efa_mr, before fi_mr_reg_impl():
if (_fi_mr_should_inject()) {
    fprintf(stderr, "[EFA_MR_INJECT] Injecting -FI_ENOMEM\n");
    free(efa_mr);
    return -FI_ENOMEM;
}
```

### Step 2: Build Two aws-ofi-nccl Variants

| Image | aws-ofi-nccl | Bug Status | Expected with Faults |
|-------|--------------|------------|----------------------|
| `scientific-with-bug` | v1.17.2 + bug reintroduced | Line 2615: `dereg_mr()` | **DEADLOCK** |
| `scientific-fixed` | v1.17.2 (unchanged) | Line 2615: `dereg_mr_no_lock()` | **SUCCESS** |

### Step 3: Run Identical Tests

Both images run with **identical fault injection settings**:

```bash
export FI_EFA_MR_INJECT_ENABLE=1
export FI_EFA_MR_INJECT_START=500   # Start after 500 MR calls
export FI_EFA_MR_INJECT_RATE=50     # Inject error every 50 calls
```

---

## Expected Results

### Test A: scientific-with-bug

```
[EFA_MR_INJECT] ENABLED: start=500 rate=50
...normal operation...
[EFA_MR_INJECT] Injecting -FI_ENOMEM at call 500
                ↓
    fi_mr_regattr() returns -FI_ENOMEM
                ↓
    reg_mr_on_device() error handler
                ↓
    this->dereg_mr(ret_handle)  ← Tries to acquire lock
                ↓
    *** DEADLOCK *** (waiting for lock we already hold)
```

**Observable symptoms:**
- Training hangs
- GPU utilization drops to 0%
- No error messages (silent hang)
- `strace` shows process stuck on `futex(FUTEX_WAIT)`

### Test B: scientific-fixed

```
[EFA_MR_INJECT] ENABLED: start=500 rate=50
...normal operation...
[EFA_MR_INJECT] Injecting -FI_ENOMEM at call 500
                ↓
    fi_mr_regattr() returns -FI_ENOMEM
                ↓
    reg_mr_on_device() error handler
                ↓
    dereg_mr_no_lock(ret_handle)  ← No lock acquisition
                ↓
    Error returned to caller → Graceful handling
                ↓
    Training continues (or graceful failure)
```

**Observable symptoms:**
- Warning message logged: "fi_mr_reg failed"
- Training either continues or fails gracefully
- No deadlock

---

## Files Created

```
aws-ofi-nccl-patches/
├── efa_mr_fault_inject.h           # Fault injection header
├── Dockerfile.scientific-fault-inject  # Builds both variants
├── pytorchjob-scientific-comparison.yaml  # K8s test manifest
└── SCIENTIFIC_PROOF_APPROACH.md    # This document
```

---

## ECR Images

```
058264135704.dkr.ecr.us-east-2.amazonaws.com/deadlock-test:
├── scientific-with-bug   # v1.17.2 + bug + fault injection
└── scientific-fixed      # v1.17.2 + fault injection (control)
```

---

## Running the Test

### Deploy Test with Bug

```bash
kubectl apply -f pytorchjob-scientific-with-bug.yaml -n kubeflow
kubectl logs -f scientific-with-bug-master-0 -n kubeflow
# Expected: Deadlock after ~500 MR calls
```

### Deploy Control Test

```bash
kubectl apply -f pytorchjob-scientific-fixed.yaml -n kubeflow
kubectl logs -f scientific-fixed-master-0 -n kubeflow
# Expected: Completes or graceful failure
```

---

## Success Criteria

| Test | Image | Fault Injection | Expected Outcome |
|------|-------|-----------------|------------------|
| A | scientific-with-bug | ✅ Enabled | **DEADLOCK** |
| B | scientific-fixed | ✅ Enabled | **Completes/Graceful** |

**If Test A deadlocks and Test B does not, the bug is scientifically proven.**

---

## Conclusion

This approach provides:

1. **Controlled experiment** - Same fault injection, different code
2. **Reproducible results** - Deterministic fault injection
3. **Clear evidence** - Deadlock vs success with identical inputs
4. **Scientific validity** - Independent variable isolation

The test proves that PR #968 fix was **necessary** because the bug causes real deadlocks when MR registration errors occur.

---

## References

- [PR #968: Fix deadlock in reg_mr() function](https://github.com/aws/aws-ofi-nccl/pull/968)
- [AWS libfabric](https://github.com/aws/libfabric)
- [Mycroft: NCCL Fault Injection (SOSP 2025)](https://arxiv.org/abs/2509.03018)
