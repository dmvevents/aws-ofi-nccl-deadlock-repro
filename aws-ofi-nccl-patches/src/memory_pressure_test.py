#!/usr/bin/env python3
"""
Memory Pressure Test for aws-ofi-nccl Deadlock Bug

This test attempts to trigger MR (Memory Registration) failures naturally
by creating memory pressure conditions. The goal is to exhaust MR resources
to cause fi_mr_regattr() to fail, which will exercise the error handling
path in aws-ofi-nccl.

The deadlock bug (fixed in PR #968) occurs when:
1. fi_mr_regattr() fails (returns error like -FI_ENOMEM)
2. Error handler calls dereg_mr() while holding mr_cache lock
3. dereg_mr() tries to acquire the same lock -> DEADLOCK

Expected behavior:
- Buggy version (v1.14.0 or reintroduced bug): DEADLOCK when MR fails
- Fixed version (v1.17.2): Graceful error handling, continues or fails cleanly
"""

import os
import sys
import time
import gc
import torch
import torch.distributed as dist

def setup_distributed():
    """Initialize distributed environment."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def allocate_gpu_memory(target_fraction=0.85):
    """
    Allocate GPU memory to create pressure.
    Returns list of tensors that should be kept alive.
    """
    device = torch.cuda.current_device()
    total_mem = torch.cuda.get_device_properties(device).total_memory
    target_bytes = int(total_mem * target_fraction)

    tensors = []
    allocated = 0
    chunk_size = 256 * 1024 * 1024  # 256MB chunks

    while allocated < target_bytes:
        try:
            t = torch.empty(chunk_size // 2, dtype=torch.float16, device='cuda')  # 2 bytes per element
            tensors.append(t)
            allocated += chunk_size
        except RuntimeError:
            break

    return tensors, allocated

def stress_mr_registration(rank, world_size, iterations=1000):
    """
    Stress memory registration by creating varying tensor sizes
    and performing collective operations that require new MR entries.
    """
    report_interval = int(os.environ.get("REPORT_INTERVAL", "100"))

    for i in range(iterations):
        # Create tensors of varying sizes to stress MR cache
        # Different sizes = different MR entries needed
        sizes = [
            (1024 + (i % 16) * 64, 1024),
            (2048 + (i % 8) * 128, 512),
            (512, 2048 + (i % 32) * 32),
        ]

        for h, w in sizes:
            try:
                t = torch.randn(h, w, device='cuda', dtype=torch.float16)

                # All-reduce requires memory registration for RDMA
                dist.all_reduce(t)

                # Immediately delete to force new allocations
                del t

            except RuntimeError as e:
                if rank == 0:
                    print(f"[Iter {i}] RuntimeError: {e}")
                # If we get here with buggy code, MR error was handled
                # (without deadlock). With deadlock, we'd never reach here.

        # Occasional all-to-all for more MR churn
        if i % 10 == 0:
            chunk_size = 1024
            input_tensor = torch.randn(chunk_size * world_size, 256, device='cuda', dtype=torch.float16)
            input_chunks = list(input_tensor.chunk(world_size))
            output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world_size)]
            dist.all_to_all(output_chunks, input_chunks)
            del input_tensor, input_chunks, output_chunks

        # Force memory cleanup to stress registration/deregistration
        if i % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        if rank == 0 and (i + 1) % report_interval == 0:
            print(f"[Progress] Iteration {i + 1}/{iterations}")

    return True

def main():
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print("=" * 70)
        print("MEMORY PRESSURE TEST FOR aws-ofi-nccl DEADLOCK BUG")
        print("=" * 70)
        print(f"World size: {world_size}")
        print(f"NCCL version: {torch.cuda.nccl.version()}")
        print()
        print("This test creates memory pressure to trigger MR allocation failures.")
        print("Expected behavior:")
        print("  - Buggy code: DEADLOCK (hang with 0% GPU activity)")
        print("  - Fixed code: Graceful handling or clean error")
        print("=" * 70)
        print()

    iterations = int(os.environ.get("ITERATIONS", "2000"))
    memory_pressure = float(os.environ.get("MEMORY_PRESSURE", "0.80"))

    # Phase 1: Allocate memory to create pressure
    if rank == 0:
        print(f"[Phase 1] Allocating {memory_pressure*100:.0f}% GPU memory...")

    held_tensors, allocated_bytes = allocate_gpu_memory(memory_pressure)
    dist.barrier()

    if rank == 0:
        print(f"[Phase 1] Allocated {allocated_bytes / (1024**3):.2f} GB per GPU")
        print()

    # Phase 2: Warmup
    if rank == 0:
        print("[Phase 2] Warmup...")

    for i in range(20):
        t = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
        dist.all_reduce(t)
        del t

    dist.barrier()
    if rank == 0:
        print("[Phase 2] Warmup complete")
        print()

    # Phase 3: Stress test
    if rank == 0:
        print(f"[Phase 3] Running {iterations} stress iterations...")
        print()

    start_time = time.time()
    success = stress_mr_registration(rank, world_size, iterations)
    elapsed = time.time() - start_time

    dist.barrier()

    # Phase 4: Cleanup and report
    del held_tensors
    torch.cuda.empty_cache()

    if rank == 0:
        print()
        print("=" * 70)
        if success:
            print("TEST COMPLETED - NO DEADLOCK DETECTED")
            print(f"Completed {iterations} iterations in {elapsed:.1f}s")
            print(f"Rate: {iterations/elapsed:.1f} iterations/sec")
        else:
            print("TEST FAILED - See errors above")
        print("=" * 70)

    dist.destroy_process_group()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        rank = int(os.environ.get("RANK", 0))
        print(f"[Rank {rank}] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
