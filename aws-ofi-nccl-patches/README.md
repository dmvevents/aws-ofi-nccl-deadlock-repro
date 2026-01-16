# aws-ofi-nccl Deadlock Bug - Scientific Proof

This repository contains the **scientific proof** that Bug #1 (MR Registration Deadlock, [PR #968](https://github.com/aws/aws-ofi-nccl/pull/968)) in aws-ofi-nccl v1.14.0 causes real production failures. This validates that the upgrade to v1.17.2 was necessary.

## Status: BUG PROVEN

| Metric | Buggy (v1.14.0) | Fixed (v1.17.2) |
|--------|-----------------|-----------------|
| After MR error | **DEADLOCK** | Error propagated |
| GPU utilization | **0% forever** | Process exits cleanly |
| Process state | Hung indefinitely | Exit code 1 |

**See [BUG_PROOF.md](BUG_PROOF.md) for complete scientific documentation.**

---

## The Bug (One Line)

In `reg_mr_on_device()` at line 2615 of `nccl_ofi_rdma.cpp`:

```diff
- (void) this->dereg_mr(ret_handle);
+ (void) dereg_mr_no_lock(ret_handle);
```

When MR registration fails, the buggy code calls `dereg_mr()` which attempts to acquire a mutex that's already held, causing a **deadlock**.

---

## Repository Structure

```
.
├── BUG_PROOF.md                    # Complete scientific proof document
├── README.md                       # This file
├── docker/                         # Dockerfiles
│   ├── Dockerfile.scientific-fault-inject  # Main test image (supports buggy/fixed)
│   └── ...
├── k8s/                            # Kubernetes manifests
│   ├── pytorchjob-fault-inject.yaml        # Buggy version test
│   ├── pytorchjob-fault-inject-fixed.yaml  # Fixed version test (control)
│   └── ...
├── src/                            # Source code
│   ├── efa_mr_fault_inject.h       # Fault injection header for libfabric
│   ├── moe_stress_test.py          # MoE workload for stress testing
│   └── ...
├── patches/                        # Patch files
│   ├── reintroduce-deadlock-bug.patch
│   └── ...
└── docs/                           # Additional documentation
    ├── APPROACH_LOG.md             # Detailed log of all approaches tried
    └── ...
```

---

## Bug Summary

| Bug | PR/Ticket | File | Description | Status |
|-----|-----------|------|-------------|--------|
| Bug₁ | PR #968 | `nccl_ofi_rdma.cpp` | MR registration deadlock | **PROVEN** |
| Bug₂ | D374802781 | `nccl_ofi_sendrecv.cpp` | Freelist leak | Fixed by EFA team |
| Bug₃ | PR #1087 | `nccl_ofi_rdma.cpp` | Schedule leak after fi_write | Needs v1.17.3 |

---

## Quick Reproduction

### Docker Images

```bash
# Buggy version (deadlocks on MR errors)
058264135704.dkr.ecr.us-east-2.amazonaws.com/deadlock-test:scientific-with-bug-v3

# Fixed version (handles errors gracefully)
058264135704.dkr.ecr.us-east-2.amazonaws.com/deadlock-test:scientific-fixed-v3
```

### Run Tests

```bash
# Run buggy version test (will DEADLOCK)
kubectl apply -f k8s/pytorchjob-fault-inject.yaml -n kubeflow

# Run fixed version test (will exit cleanly)
kubectl apply -f k8s/pytorchjob-fault-inject-fixed.yaml -n kubeflow

# Monitor logs
kubectl logs -f fault-inject-bug-test-master-0 -n kubeflow
```

### Build Images

```bash
# Build buggy version (reintroduces deadlock bug)
docker build -f docker/Dockerfile.scientific-fault-inject \
    --build-arg BUG_MODE=with-bug \
    -t deadlock-test:scientific-with-bug-v3 .

# Build fixed version (control group)
docker build -f docker/Dockerfile.scientific-fault-inject \
    --build-arg BUG_MODE=fixed \
    -t deadlock-test:scientific-fixed-v3 .
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [BUG_PROOF.md](BUG_PROOF.md) | **Complete scientific proof** with methodology and results |
| [docs/APPROACH_LOG.md](docs/APPROACH_LOG.md) | Detailed log of all 6 approaches tried |
| [docs/SCIENTIFIC_PROOF_APPROACH.md](docs/SCIENTIFIC_PROOF_APPROACH.md) | Original test methodology design |
| [docs/SCIENTIFIC_TEST_PLAN.md](docs/SCIENTIFIC_TEST_PLAN.md) | Initial test planning |

---

## Key Files

| File | Purpose |
|------|---------|
| `docker/Dockerfile.scientific-fault-inject` | Builds both buggy and fixed images |
| `src/efa_mr_fault_inject.h` | Fault injection header for libfabric |
| `src/moe_stress_test.py` | MoE workload for stress testing |
| `k8s/pytorchjob-fault-inject.yaml` | K8s manifest for buggy version |
| `k8s/pytorchjob-fault-inject-fixed.yaml` | K8s manifest for fixed version |
| `patches/reintroduce-deadlock-bug.patch` | Patch to reintroduce the bug |

---

## References

- [aws-ofi-nccl PR #968](https://github.com/aws/aws-ofi-nccl/pull/968) - The deadlock fix
- [aws-ofi-nccl v1.17.0 Release Notes](https://github.com/aws/aws-ofi-nccl/releases/tag/v1.17.0) - Release containing fix
- [AWS EFA Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html) - Elastic Fabric Adapter
