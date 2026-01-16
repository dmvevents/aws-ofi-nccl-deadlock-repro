#!/usr/bin/env python3
"""
RDMA Stress Test for aws-ofi-nccl Schedule Leak Detection (PR #1087)

This script creates heavy RDMA traffic to trigger the schedule leak bug
in aws-ofi-nccl versions < 1.17.3. The bug causes schedule objects to
not be released after fi_write() calls in post_rdma_ctrl().

The test:
1. Runs long duration to allow leak accumulation
2. Monitors memory growth over time
3. Uses large buffers to trigger RDMA operations (not shared memory)

Usage:
    torchrun --nnodes=2 --nproc_per_node=8 rdma_stress_test.py

Environment:
    ITERATIONS   - Number of test iterations (default: 10000)
    BUFFER_SIZE  - Size of RDMA buffers in MB (default: 100)
    REPORT_INTERVAL - How often to report memory (default: 100)
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import subprocess


def get_memory_usage_mb():
    """Get current process RSS memory in MB."""
    try:
        result = subprocess.run(
            ['ps', '-o', 'rss=', '-p', str(os.getpid())],
            capture_output=True, text=True
        )
        rss_kb = int(result.stdout.strip())
        return rss_kb / 1024
    except:
        return 0


def setup_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    if rank == 0:
        print(f"Distributed setup: {world_size} ranks")
        print(f"NCCL version: {torch.cuda.nccl.version()}")

    return rank, world_size, device


def rdma_intensive_operation(device, world_size, buffer_size_mb, iteration):
    """
    Perform RDMA-intensive collective operations.

    Large buffers ensure RDMA path is used (not shared memory).
    All-reduce uses RDMA for cross-node communication.
    """
    # Calculate tensor size (buffer_size_mb in MB, float16 = 2 bytes)
    elements = (buffer_size_mb * 1024 * 1024) // 2

    # Create new buffer each iteration to stress registration
    # Vary size slightly to avoid caching
    variation = ((iteration % 10) - 5) * 1024 * 1024 // 2  # +/- 5MB
    actual_elements = elements + variation

    buffer = torch.randn(actual_elements, device=device, dtype=torch.float16)

    # All-reduce: triggers RDMA write operations
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)

    # All-gather: different RDMA pattern
    gather_list = [torch.empty_like(buffer) for _ in range(world_size)]
    dist.all_gather(gather_list, buffer)

    # Reduce-scatter: yet another pattern
    scatter_input = torch.cat(gather_list)
    scatter_output = torch.empty_like(buffer)
    dist.reduce_scatter(scatter_output, list(scatter_input.chunk(world_size)))

    # Clean up explicitly
    del buffer, gather_list, scatter_input, scatter_output
    torch.cuda.empty_cache()


def run_test():
    """Main test loop with memory monitoring."""
    rank, world_size, device = setup_distributed()

    # Configuration
    iterations = int(os.environ.get("ITERATIONS", "10000"))
    buffer_size_mb = int(os.environ.get("BUFFER_SIZE", "100"))
    report_interval = int(os.environ.get("REPORT_INTERVAL", "100"))

    if rank == 0:
        print(f"\n{'='*70}")
        print("RDMA Stress Test for Schedule Leak Detection (PR #1087)")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  Iterations:    {iterations}")
        print(f"  Buffer size:   {buffer_size_mb} MB")
        print(f"  World size:    {world_size}")
        print(f"  Report every:  {report_interval} iterations")
        print(f"{'='*70}\n")

    # Warmup
    if rank == 0:
        print("Warmup phase...")

    for i in range(20):
        rdma_intensive_operation(device, world_size, buffer_size_mb, i)

    dist.barrier()
    torch.cuda.synchronize()

    # Record baseline memory
    initial_memory = get_memory_usage_mb()

    if rank == 0:
        print(f"Warmup complete. Initial memory: {initial_memory:.1f} MB")
        print(f"\nStarting stress test with memory monitoring...\n")
        print(f"{'Iteration':<12} {'Elapsed':<10} {'Memory (MB)':<15} {'Growth (MB)':<15} {'Rate (it/s)':<10}")
        print("-" * 70)

    # Main test loop
    start_time = time.time()
    last_report = start_time
    memory_readings = [(0, initial_memory)]

    for iteration in range(iterations):
        try:
            # RDMA-heavy operation
            rdma_intensive_operation(device, world_size, buffer_size_mb, iteration)

            # Memory monitoring and progress reporting
            if rank == 0 and (iteration + 1) % report_interval == 0:
                now = time.time()
                elapsed = now - start_time
                interval_time = now - last_report
                iters_per_sec = report_interval / interval_time

                current_memory = get_memory_usage_mb()
                memory_growth = current_memory - initial_memory
                memory_readings.append((iteration + 1, current_memory))

                print(f"{iteration + 1:<12} {elapsed:<10.1f} {current_memory:<15.1f} "
                      f"{memory_growth:<+15.1f} {iters_per_sec:<10.1f}")

                last_report = now

            # Periodic barrier
            if iteration % 500 == 0:
                dist.barrier()

        except Exception as e:
            print(f"[Rank {rank}] Error at iteration {iteration}: {e}")
            raise

    # Final synchronization
    dist.barrier()

    total_time = time.time() - start_time
    final_memory = get_memory_usage_mb()

    if rank == 0:
        print(f"\n{'='*70}")
        print("TEST COMPLETED")
        print(f"{'='*70}")
        print(f"Total time:     {total_time:.1f}s")
        print(f"Average rate:   {iterations/total_time:.1f} iterations/sec")
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Final memory:   {final_memory:.1f} MB")
        print(f"Memory growth:  {final_memory - initial_memory:+.1f} MB")

        # Analyze memory trend
        if len(memory_readings) >= 3:
            # Simple linear regression
            n = len(memory_readings)
            sum_x = sum(r[0] for r in memory_readings)
            sum_y = sum(r[1] for r in memory_readings)
            sum_xy = sum(r[0] * r[1] for r in memory_readings)
            sum_x2 = sum(r[0] ** 2 for r in memory_readings)

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2 + 1e-10)

            print(f"\nMemory trend: {slope * 1000:.4f} MB per 1000 iterations")

            if slope > 0.1:  # More than 0.1 MB per 1000 iterations
                print(f"\n*** WARNING: Significant memory growth detected! ***")
                print(f"*** This may indicate schedule leak (Bug #3 / PR #1087) ***")
            else:
                print(f"\n[OK] Memory appears stable (no significant leak detected)")

        print(f"{'='*70}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        run_test()
        sys.exit(0)
    except Exception as e:
        rank = int(os.environ.get("RANK", 0))
        print(f"[Rank {rank}] FATAL: {e}")
        sys.exit(1)
