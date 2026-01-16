# Hypothesis: Two Distinct Bugs in aws-ofi-nccl

This document presents the hypothesis that large-scale MoE training failures on AWS EFA were caused by **two separate bugs** in aws-ofi-nccl, not a single issue.

## The Two Bugs

### Bug #1: Memory Registration Deadlock (PR #968)

**Location:** `nccl_ofi_rdma.cpp`
**Versions Affected:** < 1.17.2
**Fix:** [PR #968](https://github.com/aws/aws-ofi-nccl/pull/968)

**Root Cause:**
When `fi_mr_reg()` fails, the mutex protecting the endpoint is not released:

```cpp
// VULNERABLE CODE
static int reg_mr(...) {
    pthread_mutex_lock(&ep->lock);

    ret = fi_mr_reg(...);
    if (ret != 0) {
        return ret;  // BUG: Lock not released!
    }

    pthread_mutex_unlock(&ep->lock);
    return 0;
}
```

**Symptoms:**
- Training hangs with 0% GPU utilization
- NCCL timeout errors
- `strace` shows processes waiting on futex
- Occurs under memory pressure or EFA resource contention

### Bug #2: Freelist Memory Leak (Separate Fix)

**Location:** `nccl_ofi_sendrecv.cpp`
**Versions Affected:** 1.17.0 - 1.17.x (introduced with Control-over-Write protocol)

**Root Cause:**
The v1.17.0 "Control-over-Write" protocol introduced a new RDMA mailbox architecture. Error paths in this new code failed to return allocated entries to the freelist:

```cpp
// LEAK: Entry allocated but not freed on error path
entry = freelist_pop(&ep->freelist);
if (some_error_condition) {
    return error;  // entry leaked!
}
```

**Symptoms:**
- Gradual memory growth over hours/days
- OOM after extended training runs
- Performance degradation before crash

## Why Two Bugs, Not One?

### Logical Argument

Consider this timeline:

1. **Before Upgrade:** Training deadlocks within hours
2. **After Upgrade (1.14.0 → 1.17.2):** Training runs stable for days
3. **After ~5 Days:** OOM crash

**If only one bug existed:**
- If the deadlock bug was the only issue, training would either deadlock OR run indefinitely
- There would be no delayed failure mode

**The delayed OOM proves two bugs:**
- Bug #1 (deadlock) was fixed by the upgrade → Training no longer deadlocks
- Bug #2 (leak) was introduced/exposed → New failure mode appears after days

### Evidence Matrix

| Observation | One Bug Hypothesis | Two Bug Hypothesis |
|-------------|-------------------|-------------------|
| Deadlock before upgrade | ✓ Explained | ✓ Explained |
| Stable training after upgrade | ✗ Unexplained (why would bug disappear?) | ✓ Deadlock fixed |
| OOM after 5 days | ✗ Unexplained | ✓ Memory leak |
| Two different failure modes | ✗ Unexplained | ✓ Two separate bugs |

### Code Path Analysis

The bugs are in completely different subsystems:

| Aspect | Bug #1 (Deadlock) | Bug #2 (Leak) |
|--------|------------------|---------------|
| File | `nccl_ofi_rdma.cpp` | `nccl_ofi_sendrecv.cpp` |
| Subsystem | Memory registration | Send/receive protocol |
| Trigger | MR registration failure | Normal operation |
| Manifestation | Immediate hang | Gradual degradation |
| Detection | Easy (hang) | Hard (slow leak) |

## Test Design

This repository provides tools to validate the deadlock bug (Bug #1) through controlled reproduction:

### Experiment Design

| Test | Library | Error Injection | Expected Result |
|------|---------|-----------------|-----------------|
| Vulnerable | 1.14.0 | Enabled | **DEADLOCK** |
| Control | 1.17.2 | Enabled | **COMPLETES** |

### Null Hypothesis

**H₀:** The upgrade from 1.14.0 to 1.17.2 does not affect deadlock behavior under MR registration stress.

### Alternative Hypothesis

**H₁:** Version 1.14.0 deadlocks under MR registration stress, while 1.17.2 recovers gracefully.

### Acceptance Criteria

- **Reject H₀ if:** Vulnerable test deadlocks AND control test completes
- **Fail to reject H₀ if:** Both tests behave identically

## Implications

If the two-bug hypothesis is validated:

1. **The upgrade to 1.17.2 was necessary** to fix the deadlock
2. **A separate fix is needed** for the memory leak (from EFA team or newer version)
3. **Both fixes together** are required for stable long-running training

## References

- [aws-ofi-nccl PR #968](https://github.com/aws/aws-ofi-nccl/pull/968) - Deadlock fix
- [aws-ofi-nccl GitHub](https://github.com/aws/aws-ofi-nccl) - Source code
- [v1.17.0 Release Notes](https://github.com/aws/aws-ofi-nccl/releases/tag/v1.17.0) - Control-over-Write protocol introduction
