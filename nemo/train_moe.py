import argparse
import gc
import os
import torch
import torch.distributed as dist
import yaml
from pathlib import Path
from collections import defaultdict
from pydantic import BaseModel, model_validator
from typing import Literal, Optional
import nemo
from nemo.collections import llm

from nemo.collections.llm.gpt.model import MoE17BMOEConfig17B, Qwen3Model
from nemo.collections.llm.gpt.model import DeepSeekV2LiteConfig, DeepSeekModel
from nemo.collections.llm.gpt.data import MockDataModule, PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

from megatron.core.optimizer import OptimizerConfig
from megatron.core.distributed import DistributedDataParallelConfig  # noqa: F401 (kept if you uncomment ddp)
from nemo.lightning.pytorch.strategies.utils import RestoreConfig  # noqa: F401

from nemo.lightning import Trainer, AutoResume, MegatronStrategy, MegatronMixedPrecision, NeMoLogger
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.pytorch.callbacks import ModelCheckpoint, GarbageCollectionCallback

from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary, Callback
from lightning.pytorch.loggers import WandbLogger
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.utils.exp_manager import TimingCallback

# =============================================================================
# ISSUE: OptimizerMonitor disabled - causes performance issues with MoE
# The callback iterates over ALL parameters every batch, which is expensive
# with 64 experts. Re-enable only for debugging specific gradient issues.
# =============================================================================
# from experimental_layer_per_param import OptimizerMonitor


# =============================================================================
# SAFE GARBAGE COLLECTION FOR DISTRIBUTED TRAINING
# =============================================================================
#
# BACKGROUND:
# Training hung at step 50, correlated with GC being triggered (see logs in
# ISSUES_FOUND.md:19-25). GarbageCollectionCallback was disabled as precaution.
#
# HYPOTHESIS (not proven):
# Unsynchronized GC could cause rank desynchronization in distributed training.
# This is a timing-dependent race condition, not guaranteed to occur.
#
# SAFE GC APPROACH:
# If GC is needed, use barriers to synchronize all ranks before/after GC.
#
# WHEN TO ENABLE SAFE GC:
# - Memory usage > 70% of GPU (approaching 56GB on H100 80GB)
# - Memory growing > 1GB per 5000 steps (leak detected)
# - Runs > 100K steps
#
# WHEN TO KEEP DISABLED (current):
# - Short runs with plenty of headroom (20K steps, ~45GB usage on 80GB GPU)
#
# HOW TO ENABLE:
# 1. Uncomment SafeGarbageCollectionCallback in callbacks list below
# 2. Optionally uncomment MemoryMonitorCallback to monitor without GC
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
    # Step 1: Wait for all GPU operations to complete
    # WHY: CUDA ops are async. GC during GPU work = memory corruption or crash
    torch.cuda.synchronize()

    # Step 2: Barrier - all ranks must arrive here before continuing
    # WHY: Prevents fast ranks from starting GC while slow ranks are in NCCL ops
    if dist.is_initialized():
        dist.barrier()

    # Step 3: Run garbage collection (all ranks together now)
    # WHY: Safe because all ranks are synchronized at this point
    gc.collect()

    # Step 4: Clear CUDA memory cache
    # WHY: gc.collect() only frees Python objects, not PyTorch's CUDA cache
    torch.cuda.empty_cache()

    # Step 5: Final barrier before continuing
    # WHY: GC duration varies by rank (depends on garbage amount).
    #      Without this, fast-GC ranks start next step while slow-GC ranks
    #      are still collecting → same deadlock problem
    if dist.is_initialized():
        dist.barrier()


class SafeGarbageCollectionCallback(Callback):
    """
    Garbage collection callback that's safe for distributed training.

    Unlike NeMo's GarbageCollectionCallback which caused hangs at step 50,
    this version only runs GC at checkpoint intervals when all ranks are
    already synchronized for checkpoint I/O.

    Args:
        gc_interval: Run GC every N steps. Should match checkpoint_interval.
        verbose: Print memory info from rank 0 before/after GC.

    Usage:
        callbacks.append(SafeGarbageCollectionCallback(gc_interval=500))

    When to enable:
        - Memory growing over time (check with MemoryMonitorCallback first)
        - Approaching GPU memory limit
        - Long runs (100K+ steps)

    When to keep disabled:
        - Short runs with plenty of memory headroom (current 20K step config)
    """

    def __init__(self, gc_interval: int = 500, verbose: bool = True):
        super().__init__()
        self.gc_interval = gc_interval
        self.verbose = verbose

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        # Only GC at checkpoint intervals
        # WHY: Checkpoints already sync all ranks, so GC here adds minimal overhead
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

    Args:
        log_interval: Log memory every N steps.
        warn_threshold_gb: Print warning if memory exceeds this (default 70GB).

    Usage:
        callbacks.append(MemoryMonitorCallback(log_interval=500))

    Example output:
        [Memory] Step 0:    42.3GB (growth: +0.00GB)
        [Memory] Step 500:  42.5GB (growth: +0.20GB)
        [Memory] Step 1000: 42.6GB (growth: +0.30GB)

    What to look for:
        - Growth < 1GB over 5000 steps → Normal, no action needed
        - Growth > 5GB over 5000 steps → Possible leak, enable SafeGC
        - Approaching 70GB → Enable SafeGC immediately
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
                    print(f"[Memory] ⚠️  WARNING: High memory usage! "
                          f"Consider enabling SafeGarbageCollectionCallback")


class NMHBlendedDatasetConfig(BaseModel):
    dataset_path: str | None = None
    dataset_prefix: str
    dataset_weight: float
    dataset_split: Literal["train", "validation", "test"]

    @model_validator(mode="before")
    @classmethod
    def validate_dataset_prefix(cls, values: dict) -> dict:
        dataset_path = Path(values.get("dataset_path")) if values.get("dataset_path") else None
        prefix = Path(values.get("dataset_prefix"))

        if not prefix.is_absolute():
            if dataset_path:
                prefix = dataset_path / prefix
            else:
                prefix = Path(prefix).resolve()
        parent = prefix.parent
        stem = prefix.stem
        if not parent.exists():
            raise ValueError(f"dataset_prefix parent path does not exist: {parent}")
        matching_files = list(parent.glob(f"{stem}.*"))
        if not matching_files:
            raise ValueError(f"dataset_prefix file does not exist: {prefix}")
        values["dataset_prefix"] = str(prefix)
        return values


def parse_dataset_config(dataset_config_path: str, dataset_path: Optional[str] = None):
    blended_dataset_config = defaultdict(list)
    weight_sums = defaultdict(float)
    with open(dataset_config_path, "r") as config_file:
        dataset_config_batch = yaml.safe_load(config_file)
        for dataset_config in dataset_config_batch:
            config_model = NMHBlendedDatasetConfig(dataset_path=dataset_path, **dataset_config)
            weight_sums[config_model.dataset_split] += abs(config_model.dataset_weight)
        for dataset_config in dataset_config_batch:
            config_model = NMHBlendedDatasetConfig(dataset_path=dataset_path, **dataset_config)
            blended_dataset_config[config_model.dataset_split].extend(
                [config_model.dataset_weight / weight_sums[config_model.dataset_split], config_model.dataset_prefix]
            )
    return blended_dataset_config


def get_auto_resume(exp_dir, restore_path=None):
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
    # Tokenizer
    tokenizer = get_nmt_tokenizer(library="huggingface", model_name="/path/to/tokenizer", use_fast=True)

    # Data (mock)
    # dataset_config  = parse_dataset_config('/path/to/data/master_config.yaml')
    # data = PreTrainingDataModule(
    #         paths=dataset_config,
    #         index_mapping_dir='/path/to/dataset_cache',
    #         tokenizer=tokenizer,
    #         seq_length=4096,
    #         micro_batch_size=2,
    #         global_batch_size=2048,
    #         seed=42,
    #         num_workers=8,
    #         #persistent_workers=True,
    #         mmap_bin_files=False
    #         )
    # ==========================================================================
    # GLOBAL BATCH SIZE CALCULATION
    # ==========================================================================
    #
    # FORMULA:
    #   global_batch_size must be divisible by (micro_batch_size × data_parallel_size)
    #   data_parallel_size = num_nodes × gpus_per_node ÷ (TP × PP)
    #
    # CURRENT CONFIG:
    #   num_nodes = 62 (64 total - 2 drained: node 19 HBM failure, node 21 down)
    #   gpus_per_node = 8
    #   TP = 1, PP = 1
    #   micro_batch_size = 2
    #
    # CALCULATION:
    #   data_parallel_size = 62 × 8 ÷ (1 × 1) = 496
    #   divisor = micro_batch_size × data_parallel_size = 2 × 496 = 992
    #   global_batch_size must be divisible by 992
    #
    # VALID BATCH SIZES FOR 62 NODES:
    #   992 × 1 = 992
    #   992 × 2 = 1984  ← CURRENT (gradient accumulation = 2)
    #   992 × 3 = 2976
    #   992 × 4 = 3968
    #
    # ==========================================================================
    # QUICK REFERENCE TABLE (micro_batch=2, TP=1, PP=1, 8 GPUs/node)
    # ==========================================================================
    #
    # | Nodes | GPUs | Divisor (2×GPUs) | Valid global_batch_size options     |
    # |-------|------|------------------|-------------------------------------|
    # | 64    | 512  | 1024             | 1024, 2048, 3072, 4096...           |
    # | 63    | 504  | 1008             | 1008, 2016, 3024, 4032...           |
    # | 62    | 496  | 992              | 992, 1984, 2976, 3968...  ← CURRENT |
    # | 61    | 488  | 976              | 976, 1952, 2928, 3904...            |
    # | 60    | 480  | 960              | 960, 1920, 2880, 3840...            |
    # | 32    | 256  | 512              | 512, 1024, 1536, 2048...            |
    # | 31    | 248  | 496              | 496, 992, 1488, 1984...             |
    # | 16    | 128  | 256              | 256, 512, 768, 1024...              |
    #
    # ==========================================================================
    # HOW TO UPDATE WHEN NODE COUNT CHANGES
    # ==========================================================================
    #
    # 1. Check available nodes:
    #    sinfo -p p5-queue
    #
    # 2. Calculate new divisor:
    #    NODES=62
    #    DIVISOR=$((2 * NODES * 8))
    #    echo "Divisor: $DIVISOR"
    #
    # 3. Choose a batch size divisible by divisor:
    #    - For similar throughput, pick closest to previous batch size
    #    - Higher = more gradient accumulation = same quality, less frequent updates
    #
    # 4. Update THREE places:
    #    a. bash.sh:  #SBATCH --nodes=XX
    #    b. main.py:  global_batch_size=XXXX (this line)
    #    c. main.py:  num_nodes=XX (in Trainer below)
    #
    # EXAMPLE - Node 19 failed, going from 64 to 63 nodes:
    #   Old: 64 nodes, divisor=1024, batch=2048
    #   New: 63 nodes, divisor=1008, batch=2016 (2048 % 1008 ≠ 0, so must change)
    #
    # EXAMPLE - Going from 64 to 62 nodes (current):
    #   Old: 64 nodes, divisor=1024, batch=2048
    #   New: 62 nodes, divisor=992, batch=1984 (closest valid to 2048)
    #
    # ==========================================================================
    data = llm.MockDataModule(seq_length=4096, global_batch_size=1984, tokenizer=tokenizer, micro_batch_size=2)

    # Auto-resume
    results_dir = "/fsx/results"
    os.makedirs(results_dir, exist_ok=True)
    auto_resume = get_auto_resume(results_dir)

    # Logging
    loggers = []
    wandb_logger = WandbLogger(
        project="17B-MoE-Production",
        name="17B-MoE-Production-run",
        save_dir=results_dir,
    )
    loggers.append(wandb_logger)
    nemo_logger = NeMoLogger(log_dir=results_dir, wandb=wandb_logger)

    # ==========================================================================
    # ISSUE FIXED: OptimizerMonitor disabled
    # The callback iterates ALL parameters every batch - very expensive with MoE
    # ==========================================================================
    # optim_callback = OptimizerMonitor()

    # Callbacks
    callbacks = [
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
        TimingCallback(),
        # ======================================================================
        # GARBAGE COLLECTION OPTIONS
        # ======================================================================
        #
        # OBSERVED ISSUE:
        # ---------------
        # Training hung at step 50. Logs showed GC was triggered immediately
        # before the hang (see ISSUES_FOUND.md:19-25 for log evidence).
        # After disabling GarbageCollectionCallback, training progressed.
        #
        # HYPOTHESIS (NOT PROVEN):
        # ------------------------
        # Unsynchronized GC may cause rank desynchronization. This is a
        # TIMING-DEPENDENT race condition - not guaranteed to occur every time.
        #
        # THEORETICAL MECHANISM:
        # In distributed training, all ranks must execute NCCL collectives
        # together. If GC runs asynchronously and one rank is delayed,
        # the collective could timeout. Example scenario:
        #
        # Rank 0: [Step 49][GC.........][Step 50]────────→ WAITING
        # Rank 1: [Step 49][    ][Step 50][AllReduce]────→ WAITING for Rank 0
        #                    ↑
        #           IF GC timing varies = POTENTIAL DEADLOCK
        #
        # ALTERNATIVE EXPLANATIONS:
        # - Thread contention during GC
        # - CUDA context issues
        # - Coincidental timing issue
        #
        # SAFE GC APPROACH (if GC needed):
        # --------------------------------
        # Rank 0: [Step 499][BARRIER][GC...][BARRIER][Step 500][AllReduce]
        # Rank 1: [Step 499][BARRIER][GC...][BARRIER][Step 500][AllReduce]
        #                     ↑                ↑
        #               All sync here    All sync here
        #
        # ======================================================================
        #
        # OPTION 1: NeMo's GarbageCollectionCallback (DISABLED)
        # Correlated with hang at step 50. Disabled as precaution.
        # GarbageCollectionCallback(gc_interval_train=50, gc_interval_val=50),
        #
        # OPTION 2: Safe GC at checkpoint intervals (USE IF MEMORY ISSUES)
        # Uncomment this if you see memory growing over time or approaching 70GB.
        # This version synchronizes all ranks before GC → no deadlock.
        # SafeGarbageCollectionCallback(gc_interval=500, verbose=True),
        #
        # OPTION 3: Memory monitoring only (USE TO DIAGNOSE)
        # Uncomment this to monitor memory without GC. Use to decide if you
        # need to enable SafeGarbageCollectionCallback.
        # MemoryMonitorCallback(log_interval=500, warn_threshold_gb=70.0),
        #
        # CURRENT STATUS: GC disabled (Option 1, 2, 3 all commented out)
        # WHY: 20K steps with ~45GB usage on 80GB H100 = plenty of headroom
        #
        # WHEN TO ENABLE SAFE GC:
        # - Memory usage > 70GB (approaching H100's 80GB limit)
        # - Memory growing > 1GB per 5000 steps (leak detected)
        # - Runs > 100K steps (garbage accumulates over time)
        # - Using real data loaders that may cache data
        #
        # WHEN TO KEEP DISABLED (CURRENT):
        # - Short runs (20K steps)
        # - Plenty of memory headroom (~35GB free)
        # - Using MockDataModule (no data caching)
        # ======================================================================
        ModelCheckpoint(
            every_n_train_steps=500,  # Reduced from 1000 to minimize loss on hardware failure
            monitor="reduced_train_loss",
            save_top_k=10,
            filename="{reduced_train_loss:.5f}-{step}-{consumed_samples}",
            dirpath=results_dir,
        ),
        # optim_callback,  # DISABLED - expensive with MoE
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

    max_steps = 20000
    warmup_steps = max_steps * 0.003
    sched = CosineAnnealingScheduler(max_steps=20000, warmup_steps=warmup_steps, min_lr=1e-5, constant_steps=0)
    optimizer_module = MegatronOptimizerModule(config=optimizer, lr_scheduler=sched)

    # Strategy & precision
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
        # ddp=DistributedDataParallelConfig(...),
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
        devices=8,
        num_nodes=62,
        max_steps=20000,
        accelerator="gpu",
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        limit_val_batches=50,
        use_distributed_sampler=False,
        plugins=precision_plugin,
        val_check_interval=1000,
    )

    # Model config
    model_config = MoE17BMOEConfig17B(moe_aux_loss_coeff=1e-3, moe_router_topk_scaling_factor=2.5)

    # Build & run
    model = Qwen3Model(config=model_config, optim=optimizer_module, tokenizer=tokenizer)

    auto_resume.setup(pl_trainer, model)
    nemo_logger.setup(pl_trainer, resume_if_exists=True)
    pl_trainer.fit(model, data)


if __name__ == "__main__":
    main()
