#!/usr/bin/env python3
"""
MoE Stress Test for aws-ofi-nccl Deadlock Reproduction

This script simulates Mixture-of-Experts (MoE) All-to-All communication patterns
that generate high memory registration churn, designed to trigger the deadlock
bug in aws-ofi-nccl versions < 1.17.2.

The test performs:
1. Distributed initialization across nodes
2. Repeated All-to-All operations with varying tensor sizes
3. Memory allocation/deallocation to stress MR registration

Usage:
    torchrun --nnodes=2 --nproc_per_node=8 moe_stress_test.py

Environment:
    ITERATIONS  - Number of test iterations (default: 1000)
    HIDDEN_DIM  - Hidden dimension size (default: 4096)
    NUM_EXPERTS - Number of experts to simulate (default: 64)
"""

import os
import sys
import time
import torch
import torch.distributed as dist


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


def simulate_moe_dispatch(rank, world_size, device, hidden_dim, num_experts, iteration):
    """
    Simulate MoE token dispatch pattern.

    This creates the All-to-All communication pattern that generates
    significant memory registration churn in aws-ofi-nccl.
    """
    # Simulate tokens being routed to different experts
    # Each rank sends different amounts to different destinations
    tokens_per_expert = hidden_dim // num_experts

    # Create input tensor (simulates tokens to dispatch)
    # Use varying sizes to stress MR registration with different buffer sizes
    size_variation = (iteration % 8) + 1
    input_size = tokens_per_expert * size_variation * num_experts

    input_tensor = torch.randn(input_size, hidden_dim, device=device, dtype=torch.float16)

    # Prepare for All-to-All
    # Split input into chunks for each destination rank
    chunk_size = input_size // world_size
    input_chunks = list(input_tensor.split(chunk_size, dim=0))

    # Pad if necessary
    while len(input_chunks) < world_size:
        input_chunks.append(torch.zeros(chunk_size, hidden_dim, device=device, dtype=torch.float16))

    # Prepare output buffers
    output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world_size)]

    # All-to-All: This is where MR registration happens
    dist.all_to_all(output_chunks, input_chunks)

    # Simulate expert computation
    combined = torch.cat(output_chunks, dim=0)
    result = combined * 1.01  # Trivial computation

    # All-to-All back (combine step)
    result_chunks = list(result.split(chunk_size, dim=0))
    while len(result_chunks) < world_size:
        result_chunks.append(torch.zeros(chunk_size, hidden_dim, device=device, dtype=torch.float16))

    final_chunks = [torch.empty_like(result_chunks[0]) for _ in range(world_size)]
    dist.all_to_all(final_chunks, result_chunks)

    return torch.cat(final_chunks, dim=0)


def stress_memory_registration(device, iteration):
    """
    Additional memory stress to increase MR registration churn.

    Creates and destroys tensors of varying sizes to trigger
    new memory registrations in the NCCL/OFI path.
    """
    sizes = [
        (1024, 1024),
        (2048, 512),
        (512, 4096),
        (4096, 256),
        (256, 8192),
    ]

    tensors = []
    for i, (h, w) in enumerate(sizes):
        # Vary sizes based on iteration
        scale = ((iteration + i) % 4) + 1
        t = torch.randn(h * scale, w, device=device, dtype=torch.float16)
        tensors.append(t)

    # Force synchronization to ensure registrations complete
    torch.cuda.synchronize()

    # Clean up (deregistration)
    del tensors
    torch.cuda.empty_cache()


def run_test():
    """Main test loop."""
    rank, world_size, device = setup_distributed()

    # Configuration
    iterations = int(os.environ.get("ITERATIONS", "1000"))
    hidden_dim = int(os.environ.get("HIDDEN_DIM", "4096"))
    num_experts = int(os.environ.get("NUM_EXPERTS", "64"))
    report_interval = int(os.environ.get("REPORT_INTERVAL", "50"))

    if rank == 0:
        print(f"\n{'='*60}")
        print("MoE Stress Test for aws-ofi-nccl Deadlock Reproduction")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Iterations:  {iterations}")
        print(f"  Hidden dim:  {hidden_dim}")
        print(f"  Num experts: {num_experts}")
        print(f"  World size:  {world_size}")
        print(f"{'='*60}\n")

    # Warmup
    if rank == 0:
        print("Warmup phase...")

    for i in range(10):
        _ = simulate_moe_dispatch(rank, world_size, device, hidden_dim, num_experts, i)
        stress_memory_registration(device, i)

    dist.barrier()
    if rank == 0:
        print("Warmup complete. Starting stress test...\n")

    # Main test loop
    start_time = time.time()
    last_report = start_time

    for iteration in range(iterations):
        try:
            # MoE dispatch simulation
            _ = simulate_moe_dispatch(rank, world_size, device, hidden_dim, num_experts, iteration)

            # Additional memory stress every few iterations
            if iteration % 5 == 0:
                stress_memory_registration(device, iteration)

            # Progress reporting
            if rank == 0 and (iteration + 1) % report_interval == 0:
                now = time.time()
                elapsed = now - start_time
                interval_time = now - last_report
                iters_per_sec = report_interval / interval_time

                print(f"[{elapsed:6.1f}s] Iteration {iteration + 1}/{iterations} "
                      f"({iters_per_sec:.1f} it/s)")
                last_report = now

            # Periodic barrier to ensure synchronization
            if iteration % 100 == 0:
                dist.barrier()

        except Exception as e:
            print(f"[Rank {rank}] Error at iteration {iteration}: {e}")
            raise

    # Final synchronization
    dist.barrier()

    total_time = time.time() - start_time
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"TEST COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average:    {iterations/total_time:.1f} iterations/sec")
        print(f"{'='*60}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        run_test()
        sys.exit(0)
    except Exception as e:
        rank = int(os.environ.get("RANK", 0))
        print(f"[Rank {rank}] FATAL: {e}")
        sys.exit(1)
