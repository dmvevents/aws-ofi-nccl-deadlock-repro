# Code Analysis: aws-ofi-nccl Deadlock Bug

This document provides a detailed analysis of the deadlock bug in aws-ofi-nccl versions prior to 1.17.2.

## Bug Location

**File:** `src/nccl_ofi_rdma.cpp`
**Function:** `reg_mr_base_comm()` and related memory registration functions

## The Vulnerable Code Pattern

In aws-ofi-nccl versions 1.14.x and earlier, the memory registration path had a critical bug:

```cpp
// Simplified representation of the vulnerable code pattern
static int reg_mr(nccl_net_ofi_rdma_ep_t *ep, void *data, size_t size,
                  int type, nccl_net_ofi_rdma_mr_handle_t **mhandle) {

    // Lock acquired to protect endpoint state
    pthread_mutex_lock(&ep->lock);

    // Attempt memory registration with libfabric
    ret = fi_mr_reg(domain->domain, data, size, access,
                    0, 0, 0, &mr_handle->mr, NULL);

    if (ret != 0) {
        // ERROR PATH - BUG IS HERE
        NCCL_OFI_WARN("Unable to register memory (fi_mr_reg failed)");
        // Missing: pthread_mutex_unlock(&ep->lock);
        return ncclSystemError;  // Lock is still held!
    }

    // Success path - lock properly released
    pthread_mutex_unlock(&ep->lock);
    return ncclSuccess;
}
```

## Why This Causes Deadlock

### Scenario

1. **Thread A** calls `reg_mr()`, acquires the mutex
2. `fi_mr_reg()` fails due to:
   - Memory pressure
   - EFA resource exhaustion
   - Transient network error
3. **Thread A** returns error without releasing mutex
4. **Thread B** (or any other thread) calls `reg_mr()`
5. **Thread B** blocks on `pthread_mutex_lock(&ep->lock)` forever
6. **All NCCL operations on this endpoint are now blocked**

### Deadlock Diagram

```
Thread A                          Thread B
────────                          ────────
mutex_lock(&ep->lock) ───────┐
fi_mr_reg() → FAIL           │
return (mutex still held) ◄──┘
                              ┌──► mutex_lock(&ep->lock)
                              │    [BLOCKED FOREVER]
                              │
    Thread A exits            │
    or continues other work   │
                              │
                              ▼
                         DEADLOCK
```

## The Fix (PR #968)

The fix ensures the mutex is always released, regardless of the return path:

```cpp
// Fixed code pattern
static int reg_mr(nccl_net_ofi_rdma_ep_t *ep, void *data, size_t size,
                  int type, nccl_net_ofi_rdma_mr_handle_t **mhandle) {

    pthread_mutex_lock(&ep->lock);

    ret = fi_mr_reg(domain->domain, data, size, access,
                    0, 0, 0, &mr_handle->mr, NULL);

    if (ret != 0) {
        NCCL_OFI_WARN("Unable to register memory (fi_mr_reg failed)");
        pthread_mutex_unlock(&ep->lock);  // FIX: Release lock on error
        return ncclSystemError;
    }

    pthread_mutex_unlock(&ep->lock);
    return ncclSuccess;
}
```

## When Does fi_mr_reg() Fail?

Memory registration can fail under several conditions:

### 1. Resource Exhaustion

EFA has limits on:
- Maximum memory regions per process
- Total registered memory size
- Per-QP registration limits

```
$ fi_info -p efa
    max_mr: 524288
    max_mr_size: 18446744073709551615
```

### 2. Memory Pressure

When system memory is low:
- Kernel may fail to pin pages
- IOMMU mapping may fail
- DMA buffer allocation may fail

### 3. High Churn Workloads

Mixture-of-Experts (MoE) models create high MR churn:
- All-to-All communication requires many small buffers
- Buffers are created/destroyed frequently
- Parallel expert computation amplifies registration rate

```
MoE Dispatch:
  Token routing → N separate buffers → N registrations
  Expert compute → N result buffers → N more registrations
  Combine → All-to-All → More registrations
```

### 4. Transient Errors

Network or driver issues:
- EFA device reset
- PCIe errors
- Driver bugs

## Impact Analysis

### Affected Operations

Any NCCL operation that triggers memory registration:
- `ncclAllReduce` (large tensors)
- `ncclAllToAll` (used by MoE)
- `ncclBroadcast`
- `ncclReduceScatter`

### Failure Cascade

```
fi_mr_reg fails
    ↓
Mutex held
    ↓
Next MR operation blocks
    ↓
NCCL collective hangs
    ↓
Other ranks timeout waiting
    ↓
Entire distributed job fails
```

## Detection

### Symptoms

1. **GPU Utilization:** Drops to 0%
2. **CPU:** High utilization (spin waiting)
3. **strace:** Shows futex wait
4. **NCCL Logs:** Timeout after NCCL_TIMEOUT seconds

### Diagnostic Commands

```bash
# Check for blocked processes
strace -p <PID> 2>&1 | grep futex

# Check NCCL state
export NCCL_DEBUG=INFO
# Look for: "Waiting for receive"

# Check GPU activity
nvidia-smi --query-compute-apps=pid,used_memory --format=csv -l 1
```

## Verification Test

This repository provides a test to verify the bug:

1. **Build container** with both library versions
2. **Inject errors** via LD_PRELOAD to simulate fi_mr_reg failures
3. **Run MoE workload** to generate MR churn
4. **Compare:**
   - Vulnerable (1.14.0): Should deadlock
   - Fixed (1.17.2): Should recover

## Related Code Paths

Other functions that were also fixed in PR #968:

```cpp
// All of these had similar mutex leak issues:
- reg_mr_base_comm()
- dereg_mr_base_comm()
- alloc_rdma_write_handle()
- free_rdma_write_handle()
```

## References

- [PR #968: Fix mutex release on error](https://github.com/aws/aws-ofi-nccl/pull/968)
- [libfabric fi_mr_reg documentation](https://ofiwg.github.io/libfabric/v1.20.0/man/fi_mr.3.html)
- [NCCL Troubleshooting Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
