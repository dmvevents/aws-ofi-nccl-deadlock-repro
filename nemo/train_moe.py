#!/usr/bin/env python3
"""
MoE17B 17B MoE Training Script

This script trains a 17B parameter Mixture-of-Experts model using NVIDIA NeMo
and Megatron-LM. It includes distributed training optimizations and safe garbage
collection patterns discovered during production training.

The model architecture:
- Base: Qwen3-style transformer
- 17B total parameters
- 64 experts with top-2 routing
- Expert parallelism across nodes

This script is designed to stress-test the NCCL/EFA communication path,
particularly the All-to-All operations used by MoE expert routing.

Usage:
    # Via SLURM (see slurm/train_moe.sh)
    sbatch slurm/train_moe.sh

    # Direct (single node testing)
    torchrun --nproc_per_node=8 nemo/train_moe.py

Requirements:
    - NVIDIA NeMo container (nvcr.io/nvidia/nemo:25.09.00 or later)
    - AWS EFA networking
    - aws-ofi-nccl plugin (1.17.2+ recommended)
"""

import gc
import os
import torch
import torch.distributed as dist
from pathlib import Path

# NeMo imports
import nemo
from nemo.collections import llm
from nemo.collections.llm.gpt.model import MoE17BMOEConfig17B, Qwen3Model
from nemo.collections.llm.gpt.data import MockDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

# Megatron imports
from megatron.core.optimizer import OptimizerConfig

# Lightning/NeMo training imports
from nemo.lightning import Trainer, AutoResume, MegatronStrategy, MegatronMixedPrecision, NeMoLogger
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.pytorch.callbacks import ModelCheckpoint

from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary, Callback
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.utils.exp_manager import TimingCallback


# =============================================================================
# SAFE GARBAGE COLLECTION FOR DISTRIBUTED TRAINING
# =============================================================================
#
# BACKGROUND:
# Training can hang when garbage collection runs unsynchronized across ranks.
# This was observed at step 50 when GC triggered on some ranks but not others,
# causing NCCL collective timeouts.
#
# SOLUTION:
# Synchronize all ranks before and after GC using barriers.
# =============================================================================

def safe_garbage_collect():
    """
    Run garbage collection safely in distributed training.
    All ranks synchronize before and after GC to prevent deadlocks.

    Timeline WITHOUT this (HANGS):
        Rank 0: [Train 49][GC.......][Train 50]──────→ WAITING
        Rank 1: [Train 49][   ][Train 50][AllReduce]─→ WAITING for Rank 0
        Result: DEADLOCK

    Timeline WITH this (WORKS):
        Rank 0: [Train 49][BARRIER][GC...][BARRIER][Train 50][AllReduce]
        Rank 1: [Train 49][BARRIER][GC...][BARRIER][Train 50][AllReduce]
        Result: All ranks GC together, no deadlock
    """
    # Wait for all GPU operations to complete
    torch.cuda.synchronize()

    # Barrier - all ranks must arrive here before continuing
    if dist.is_initialized():
        dist.barrier()

    # Run garbage collection (all ranks together now)
    gc.collect()

    # Clear CUDA memory cache
    torch.cuda.empty_cache()

    # Final barrier before continuing
    if dist.is_initialized():
        dist.barrier()


class SafeGarbageCollectionCallback(Callback):
    """
    Garbage collection callback that's safe for distributed training.

    Args:
        gc_interval: Run GC every N steps. Should match checkpoint_interval.
        verbose: Print memory info from rank 0 before/after GC.

    Usage:
        callbacks.append(SafeGarbageCollectionCallback(gc_interval=500))
    """

    def __init__(self, gc_interval: int = 500, verbose: bool = True):
        super().__init__()
        self.gc_interval = gc_interval
        self.verbose = verbose

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        if global_step > 0 and global_step % self.gc_interval == 0:
            if self.verbose and trainer.global_rank == 0:
                before = torch.cuda.memory_allocated() / 1e9
                print(f"[SafeGC] Step {global_step}: Running synchronized GC...")

            safe_garbage_collect()

            if self.verbose and trainer.global_rank == 0:
                after = torch.cuda.memory_allocated() / 1e9
                freed = before - after
                print(f"[SafeGC] Step {global_step}: Freed {freed:.2f}GB, "
                      f"Now using {after:.1f}GB")


class MemoryMonitorCallback(Callback):
    """
    Monitor GPU memory without running GC.
    Use this to decide if you need to enable SafeGarbageCollectionCallback.
    """

    def __init__(self, log_interval: int = 500, warn_threshold_gb: float = 70.0):
        super().__init__()
        self.log_interval = log_interval
        self.warn_threshold_gb = warn_threshold_gb
        self.baseline = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_interval == 0:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9

            if self.baseline is None:
                self.baseline = allocated

            growth = allocated - self.baseline

            if trainer.global_rank == 0:
                print(f"[Memory] Step {trainer.global_step}: "
                      f"{allocated:.1f}GB allocated, {reserved:.1f}GB reserved "
                      f"(growth: {growth:+.2f}GB)")

                if allocated > self.warn_threshold_gb:
                    print(f"[Memory] WARNING: High memory usage!")


def get_auto_resume(exp_dir, restore_path=None):
    """Configure automatic checkpoint resumption."""
    restore_cfg = None
    if restore_path:
        from nemo.lightning.pytorch.strategies.utils import RestoreConfig
        restore_cfg = RestoreConfig(path=restore_path)
    return AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_from_directory=exp_dir,
        restore_config=restore_cfg,
    )


def main():
    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
    # Update these for your environment
    NUM_NODES = int(os.environ.get("NUM_NODES", "2"))
    GPUS_PER_NODE = int(os.environ.get("GPUS_PER_NODE", "8"))
    MAX_STEPS = int(os.environ.get("MAX_STEPS", "1000"))
    RESULTS_DIR = os.environ.get("RESULTS_DIR", "/tmp/moe17b_results")

    # ==========================================================================
    # GLOBAL BATCH SIZE CALCULATION
    # ==========================================================================
    #
    # FORMULA:
    #   global_batch_size must be divisible by (micro_batch_size × data_parallel_size)
    #   data_parallel_size = num_nodes × gpus_per_node ÷ (TP × PP)
    #
    # With TP=1, PP=1, EP=8:
    #   data_parallel_size = NUM_NODES × GPUS_PER_NODE
    #   micro_batch_size = 2
    #   divisor = 2 × NUM_NODES × GPUS_PER_NODE
    #
    # ==========================================================================
    micro_batch_size = 2
    data_parallel_size = NUM_NODES * GPUS_PER_NODE
    divisor = micro_batch_size * data_parallel_size

    # Choose a global batch size that's divisible by the divisor
    # For small tests, use the minimum valid value
    global_batch_size = divisor  # Minimum valid batch size

    print(f"Configuration:")
    print(f"  NUM_NODES: {NUM_NODES}")
    print(f"  GPUS_PER_NODE: {GPUS_PER_NODE}")
    print(f"  data_parallel_size: {data_parallel_size}")
    print(f"  micro_batch_size: {micro_batch_size}")
    print(f"  global_batch_size: {global_batch_size}")
    print(f"  MAX_STEPS: {MAX_STEPS}")
    print(f"  RESULTS_DIR: {RESULTS_DIR}")

    # Tokenizer (use a default HF tokenizer for testing)
    # In production, use your trained tokenizer
    tokenizer_name = os.environ.get("TOKENIZER_PATH", "Qwen/Qwen2-7B")
    try:
        tokenizer = get_nmt_tokenizer(
            library="huggingface",
            model_name=tokenizer_name,
            use_fast=True
        )
    except Exception as e:
        print(f"Warning: Could not load tokenizer {tokenizer_name}: {e}")
        print("Using mock tokenizer...")
        tokenizer = None

    # Data (mock for testing - use real data in production)
    data = llm.MockDataModule(
        seq_length=4096,
        global_batch_size=global_batch_size,
        tokenizer=tokenizer,
        micro_batch_size=micro_batch_size
    )

    # Results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    auto_resume = get_auto_resume(RESULTS_DIR)

    # Logging
    nemo_logger = NeMoLogger(log_dir=RESULTS_DIR)

    # Callbacks
    callbacks = [
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
        TimingCallback(),
        # Memory monitoring (enable to track memory usage)
        MemoryMonitorCallback(log_interval=100, warn_threshold_gb=70.0),
        # Safe GC (enable if memory issues detected)
        # SafeGarbageCollectionCallback(gc_interval=500, verbose=True),
        ModelCheckpoint(
            every_n_train_steps=500,
            monitor="reduced_train_loss",
            save_top_k=3,
            filename="{reduced_train_loss:.5f}-{step}-{consumed_samples}",
            dirpath=RESULTS_DIR,
        ),
    ]

    # Optimizer & scheduler
    optimizer = OptimizerConfig(
        optimizer="adam",
        lr=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        clip_grad=1.0,
        use_distributed_optimizer=True,
        bf16=True,
    )

    warmup_steps = int(MAX_STEPS * 0.003)
    sched = CosineAnnealingScheduler(
        max_steps=MAX_STEPS,
        warmup_steps=warmup_steps,
        min_lr=1e-5,
        constant_steps=0
    )
    optimizer_module = MegatronOptimizerModule(config=optimizer, lr_scheduler=sched)

    # Strategy & precision
    # Expert parallelism = 8 means experts are distributed across 8 GPUs
    # With 64 experts and EP=8, each GPU handles 8 experts
    strategy = MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=8,
        context_parallel_size=1,
        sequence_parallel=False,
        pipeline_dtype=torch.bfloat16,
        ckpt_async_save=False,
        ckpt_load_strictness="log_all",
        gradient_as_bucket_view=True,
    )

    precision_plugin = MegatronMixedPrecision(
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )

    # Trainer
    pl_trainer = Trainer(
        devices=GPUS_PER_NODE,
        num_nodes=NUM_NODES,
        max_steps=MAX_STEPS,
        accelerator="gpu",
        strategy=strategy,
        callbacks=callbacks,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        limit_val_batches=50,
        use_distributed_sampler=False,
        plugins=precision_plugin,
        val_check_interval=500,
    )

    # Model config - MoE17B 17B MoE
    # This creates a model with:
    # - ~17B parameters
    # - 64 experts
    # - Top-2 routing
    model_config = MoE17BMOEConfig17B(
        moe_aux_loss_coeff=1e-3,
        moe_router_topk_scaling_factor=2.5
    )

    # Build & run
    model = Qwen3Model(
        config=model_config,
        optim=optimizer_module,
        tokenizer=tokenizer
    )

    auto_resume.setup(pl_trainer, model)
    nemo_logger.setup(pl_trainer, resume_if_exists=True)
    pl_trainer.fit(model, data)


if __name__ == "__main__":
    main()
