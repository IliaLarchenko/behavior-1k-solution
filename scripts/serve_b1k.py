import dataclasses
import enum
import logging
import os
import pathlib
import socket

import numpy as np
import tyro

# Set JAX memory allocation before importing JAX (can be overridden by env vars)
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.5')  # Use 50% of GPU memory
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')  # Platform allocator

from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
from omnigibson.learning.datas import BehaviorLerobotDatasetMetadata

from openpi.policies import policy as _policy

# Import B1K-specific modules
from b1k.policies import policy_config as _policy_config  # Use our custom policy_config
from b1k.policies.checkpoint_switcher import CheckpointSwitcher
from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, B1KWrapperConfig
from b1k.training import config as _config


class EnvMode(enum.Enum):
    # Not used, just kept for compatibility
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default prompt.
    default_prompt: str | None = None
    
    # For PI_BEHAVIOR models: task ID (0-49) instead of text prompt
    task_id: int | None = None

    # Dataset root, used to retrieve the prompt of the task if taskname is not None.
    dataset_root: str | None = "/scr/behavior/2025-challenge-demos"
    # If provided, will be used to retrieve the prompt of the task, otherwise use turning_on_radio as default.
    task_name: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    
    # B1K Wrapper execution parameters
    actions_to_execute: int = 26
    actions_to_keep: int = 4
    execute_in_n_steps: int = 20
    history_len: int = 3
    votes_to_promote: int = 2
    time_threshold_inpaint: float = 0.3
    num_steps: int = 20
    apply_eval_tricks: bool = True  # Enable correction rules and gripper variation checks
    
    # Multi-checkpoint support for PI_BEHAVIOR models (optional)
    task_checkpoint_mapping: str | None = None  # Path to task-checkpoint mapping JSON file


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    sample_kwargs = {"num_steps": args.num_steps}
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config), 
        args.policy.dir, 
        default_prompt=args.default_prompt,
        sample_kwargs=sample_kwargs
    )


def main(args: Args) -> None:
    # B1K only supports PI_BEHAVIOR models (task embeddings, no text prompts)
    config = _config.get_config(args.policy.config)
    
    # PI_BEHAVIOR model setup
    if args.task_id is not None:
        logging.info(f"Using PI_BEHAVIOR model with task_id: {args.task_id}")
        task_id = args.task_id
    else:
        logging.info(f"Using PI_BEHAVIOR model - task_id will be extracted from observations")
        task_id = None
    
    # Placeholder prompt for PI_BEHAVIOR (not actually used by model)
    prompt = "PI_BEHAVIOR model (task-conditioned)"
    logging.info(f"Using prompt: {prompt}")

    # Load initial/default policy
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Create checkpoint switcher if mapping file provided
    checkpoint_switcher = None
    if args.task_checkpoint_mapping:
        logging.info(f"Multi-checkpoint mode enabled: {args.task_checkpoint_mapping}")
        
        sample_kwargs = {"num_steps": args.num_steps}
        
        try:
            checkpoint_switcher = CheckpointSwitcher(
                config_path=args.task_checkpoint_mapping,
                training_config=config,
                sample_kwargs=sample_kwargs
            )
            logging.info("Checkpoint switcher initialized - will switch checkpoints based on task_id")
        except Exception as e:
            logging.error(f"Failed to initialize checkpoint switcher: {e}")
            raise
    else:
        logging.info("Single checkpoint mode - using one checkpoint for all tasks")

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # Create wrapper configuration
    wrapper_config = B1KWrapperConfig(
        actions_to_execute=args.actions_to_execute,
        actions_to_keep=args.actions_to_keep,
        execute_in_n_steps=args.execute_in_n_steps,
        history_len=args.history_len,
        votes_to_promote=args.votes_to_promote,
        time_threshold_inpaint=args.time_threshold_inpaint,
        num_steps=args.num_steps,
        apply_eval_tricks=args.apply_eval_tricks,
    )
    
    logging.info(f"Wrapper config: execute={wrapper_config.actions_to_execute}, keep={wrapper_config.actions_to_keep}, steps={wrapper_config.execute_in_n_steps}, num_steps={wrapper_config.num_steps}")
    
    if wrapper_config.apply_eval_tricks:
        logging.info("Eval tricks ENABLED - correction rules and gripper variation checks active")
    else:
        logging.info("Eval tricks DISABLED (default behavior)")

    # Create B1K wrapper with PI_BEHAVIOR-specific features
    policy = B1KPolicyWrapper(
        policy, 
        text_prompt=prompt,  # Not used by PI_BEHAVIOR, kept for compatibility
        task_id=task_id,
        config=wrapper_config,
        checkpoint_switcher=checkpoint_switcher
    )
    
    if checkpoint_switcher:
        logging.info("Multi-checkpoint mode: checkpoints will switch based on task_id from observations")
    else:
        logging.info("Rolling inpainting enabled: will use initial_actions from input batch when provided")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
