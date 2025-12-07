"""Policy configuration for B1K - loads checkpoints and creates policies.

Exact copy of openpi.policies.policy_config but imports b1k.models.pi_behavior.PiBehavior.
"""

import logging
import os
import pathlib
from typing import Any

import numpy as np
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
import openpi.transforms as transforms

# Import B1K-specific modules
from b1k.models.pi_behavior import PiBehavior
from b1k.policies.pi_behavior_policy import PiBehaviorPolicy
from b1k.training import checkpoints as _checkpoints
from b1k.training import config as _config
from b1k import transforms as b1k_transforms
from b1k.transforms_normalize import NormalizeWithPerTimestamp, UnnormalizeWithPerTimestamp


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint - EXACT COPY from openpi with b1k imports."""
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Detect PyTorch model
    is_pytorch = (checkpoint_dir / "pytorch_model.safetensors").exists() or (checkpoint_dir / "pytorch_model.pt").exists()
    
    if is_pytorch:
        raise NotImplementedError("PyTorch inference not supported in b1k")
    
    # JAX model loading - load directly as bfloat16 to save memory (12GB vs 24GB)
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    
    # Get data config
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    
    # Load norm stats if not provided
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    
    # Load correlation matrix for PiBehavior models
    if isinstance(model, PiBehavior):
        if norm_stats is None:
            raise ValueError("PiBehavior requires norm_stats but none found.")
        model.load_correlation_matrix(norm_stats)
        logging.info("Loaded correlation matrix for inference")
    
    # Determine the device for PyTorch (not used for b1k but kept for compatibility)
    if is_pytorch and pytorch_device is None:
        try:
            import torch
            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"
    
    # For PI_BEHAVIOR models during inference, skip training-specific transforms
    model_transforms_inputs = []
    for transform in data_config.model_transforms.inputs:
        # Skip training-specific transforms during inference
        if isinstance(transform, (b1k_transforms.ComputeSubtaskStateFromMeta, b1k_transforms.TaskIndexToTaskId, b1k_transforms.TokenizeFASTActions)):
            continue
        model_transforms_inputs.append(transform)
    
    # Build input transform pipeline (skip data_config.repack_transforms - has 'actions' mapping for training)
    input_transforms = [
        *repack_transforms.inputs,
        transforms.InjectDefaultPrompt(default_prompt),
        *data_config.data_transforms.inputs,
        NormalizeWithPerTimestamp(norm_stats, use_quantiles=data_config.use_quantile_norm, use_per_timestamp=data_config.use_per_timestamp_norm),
        *model_transforms_inputs,
    ]
    
    # Build output transform pipeline
    output_transforms = [
        *data_config.model_transforms.outputs,
        UnnormalizeWithPerTimestamp(norm_stats, use_quantiles=data_config.use_quantile_norm, use_per_timestamp=data_config.use_per_timestamp_norm),
        *data_config.data_transforms.outputs,
        *repack_transforms.outputs,
    ]
    
    # Use custom PiBehaviorPolicy for PiBehavior models (handles tuple unpacking)
    if isinstance(model, PiBehavior):
        return PiBehaviorPolicy(
            model,
            transforms=input_transforms,
            output_transforms=output_transforms,
            sample_kwargs=sample_kwargs,
            metadata=train_config.policy_metadata,
        )
    else:
        return _policy.Policy(
            model,
            transforms=input_transforms,
            output_transforms=output_transforms,
            sample_kwargs=sample_kwargs,
            metadata=train_config.policy_metadata,
            is_pytorch=is_pytorch,
            pytorch_device=pytorch_device if is_pytorch else "cpu",
        )

