#!/usr/bin/env python3
"""
Single-Node NCCL Test with EFA for Fault Injection Validation

This test runs NCCL operations on a single node with 8 GPUs using EFA.
It should trigger MR registration through the aws-ofi-nccl plugin,
which will exercise the fault injection code in the patched libfabric.

Expected behavior:
- With buggy aws-ofi-nccl: DEADLOCK when fault is injected
- With fixed aws-ofi-nccl: Graceful error handling, test continues
"""

import os
import sys
import time
import torch
import torch.distributed as dist


def main():
    # Initialize distributed with NCCL backend
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    if rank == 0:
        print("=" * 60)
        print("SINGLE-NODE NCCL TEST WITH FAULT INJECTION")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"NCCL version: {torch.cuda.nccl.version()}")
        print(f"FI_EFA_MR_INJECT_ENABLE: {os.environ.get('FI_EFA_MR_INJECT_ENABLE', 'not set')}")
        print(f"FI_EFA_MR_INJECT_START: {os.environ.get('FI_EFA_MR_INJECT_START', 'not set')}")
        print(f"FI_EFA_MR_INJECT_RATE: {os.environ.get('FI_EFA_MR_INJECT_RATE', 'not set')}")
        print("=" * 60)
        print()

    # Configuration
    iterations = int(os.environ.get("ITERATIONS", "1000"))
    report_interval = int(os.environ.get("REPORT_INTERVAL", "100"))

    if rank == 0:
        print(f"Running {iterations} iterations...")
        print()

    # Warmup
    for i in range(10):
        t = torch.randn(1024, 1024, device=device, dtype=torch.float16)
        dist.all_reduce(t)
    dist.barrier()

    if rank == 0:
        print("Warmup complete. Starting test...")
        print()

    # Main test loop
    start_time = time.time()

    for i in range(iterations):
        # Varying tensor sizes to stress MR registration
        size = 1024 + (i % 8) * 256
        t = torch.randn(size, size, device=device, dtype=torch.float16)

        # All-reduce operation - triggers NCCL/EFA communication
        dist.all_reduce(t)

        # Additional operations to increase MR churn
        if i % 5 == 0:
            # All-gather
            output_list = [torch.zeros_like(t) for _ in range(world_size)]
            dist.all_gather(output_list, t)
            del output_list

        if i % 10 == 0:
            # Broadcast
            dist.broadcast(t, src=0)

        # Clean up to force new allocations
        del t
        torch.cuda.empty_cache()

        # Progress report
        if rank == 0 and (i + 1) % report_interval == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"[{elapsed:6.1f}s] Iteration {i+1}/{iterations} ({rate:.1f} it/s)")

    dist.barrier()

    total_time = time.time() - start_time
    if rank == 0:
        print()
        print("=" * 60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total time: {total_time:.1f}s")
        print(f"Average: {iterations/total_time:.1f} iterations/sec")
        print()
        print("RESULT: No deadlock detected")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        rank = int(os.environ.get("RANK", 0))
        print(f"[Rank {rank}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
