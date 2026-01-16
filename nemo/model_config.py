"""
MoE17B 17B MoE Model Configuration

This module defines the model configuration for the MoE17B 17B parameter
Mixture-of-Experts model based on Qwen3 architecture.

Model Architecture:
- 17B total parameters
- 21 layers (1 dense + 20 MoE layers)
- 64 experts with top-6 routing
- Hidden size: 2048
- FFN hidden size: 9216
- MoE FFN hidden size: 2048 per expert

Expert Parallelism Notes:
- num_moe_experts (64) must be divisible by expert_model_parallel_size
- Valid EP values: 1, 2, 4, 8, 16, 32, 64
- Recommended: EP=8 (8 experts per GPU in a node)
"""

from dataclasses import dataclass, field
from typing import Union, List

# This would be imported from NeMo in a real deployment
# from nemo.collections.llm.gpt.model import Qwen3MoEConfig


@dataclass
class Qwen3MoEConfig:
    """
    Base Qwen3 MoE configuration.

    This is a simplified version - in production, import from NeMo:
    from nemo.collections.llm.gpt.model import Qwen3MoEConfig
    """
    # Transformer architecture
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8
    ffn_hidden_size: int = 11008

    # MoE configuration
    moe_layer_freq: Union[int, List[int]] = 1
    moe_ffn_hidden_size: int = 1408
    num_moe_experts: int = 64
    moe_router_topk: int = 2
    moe_router_enable_expert_bias: bool = False
    moe_grouped_gemm: bool = True
    moe_aux_loss_coeff: float = 0.01
    moe_router_topk_scaling_factor: float = 1.0

    # Additional settings
    kv_channels: int = None
    rotary_interleaved: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0


@dataclass
class MoE17BMOEConfig17B(Qwen3MoEConfig):
    """
    MoE17B 17B MoE Configuration.

    This configuration defines a 17B parameter MoE model with:
    - 21 layers total
    - First layer is dense (no MoE)
    - Layers 1-20 are MoE layers
    - 64 experts per MoE layer
    - Top-6 routing (6 experts active per token)

    The model uses Qwen3 architecture as the base transformer.

    Expert Parallelism Calculation:
    - num_moe_experts = 64
    - Valid expert_model_parallel_size values: 1, 2, 4, 8, 16, 32, 64
    - With EP=8: 64 / 8 = 8 experts per GPU

    Memory Estimation (per GPU with EP=8):
    - Each expert FFN: 2048 * 2048 * 2 (up + down) * 2 bytes (bf16) = ~16MB
    - 8 experts per GPU: ~128MB per MoE layer
    - 20 MoE layers: ~2.5GB for expert weights alone

    All-to-All Communication Pattern:
    - Token dispatch: Each GPU sends tokens to 6 different expert GPUs (top-6)
    - Expert compute: Each GPU processes tokens routed to its experts
    - Token combine: Results gathered back to original GPUs
    - This creates O(world_size) communication per MoE layer
    """
    # Reduced model size compared to larger variants
    num_layers: int = 21
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_query_groups: int = 8
    ffn_hidden_size: int = 9216

    # MoE Configuration
    # First layer (index 0) is dense, remaining 20 layers are MoE
    moe_layer_freq: Union[int, List[int]] = field(
        default_factory=lambda: [0] + [1] * 20
    )
    moe_ffn_hidden_size: int = 2048
    num_moe_experts: int = 64  # Must be divisible by expert_model_parallel_size
    moe_router_topk: int = 6   # 6 experts active per token
    moe_router_enable_expert_bias: bool = True
    moe_grouped_gemm: bool = True

    # Auxiliary loss coefficient for load balancing
    # Higher values = more balanced expert utilization, but may hurt quality
    moe_aux_loss_coeff: float = 1e-3

    # Router scaling factor
    moe_router_topk_scaling_factor: float = 2.5


# Verification function
def verify_config_for_ep(config: MoE17BMOEConfig17B, expert_parallel_size: int) -> bool:
    """
    Verify that the model configuration is compatible with the given
    expert parallelism setting.

    Args:
        config: Model configuration
        expert_parallel_size: Number of GPUs for expert parallelism

    Returns:
        True if configuration is valid

    Raises:
        ValueError if num_moe_experts is not divisible by expert_parallel_size
    """
    if config.num_moe_experts % expert_parallel_size != 0:
        raise ValueError(
            f"num_moe_experts ({config.num_moe_experts}) must be divisible by "
            f"expert_model_parallel_size ({expert_parallel_size}). "
            f"Valid EP values: {[i for i in [1, 2, 4, 8, 16, 32, 64] if config.num_moe_experts % i == 0]}"
        )

    experts_per_gpu = config.num_moe_experts // expert_parallel_size
    print(f"Configuration valid:")
    print(f"  num_moe_experts: {config.num_moe_experts}")
    print(f"  expert_parallel_size: {expert_parallel_size}")
    print(f"  experts_per_gpu: {experts_per_gpu}")

    return True


if __name__ == "__main__":
    # Test configuration
    config = MoE17BMOEConfig17B()

    print("MoE17B 17B MoE Configuration:")
    print(f"  num_layers: {config.num_layers}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  ffn_hidden_size: {config.ffn_hidden_size}")
    print(f"  num_moe_experts: {config.num_moe_experts}")
    print(f"  moe_router_topk: {config.moe_router_topk}")
    print(f"  moe_layer_freq: {config.moe_layer_freq}")
    print()

    # Verify with different EP settings
    for ep in [1, 2, 4, 8, 16, 32, 64]:
        try:
            verify_config_for_ep(config, ep)
        except ValueError as e:
            print(f"EP={ep}: INVALID - {e}")
