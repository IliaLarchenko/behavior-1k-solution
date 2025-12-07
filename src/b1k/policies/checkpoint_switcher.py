"""Multi-task checkpoint switching for BEHAVIOR-1K.

Was used in the final submission to allow changing checkpoints based on task_id received from the server.
"""

import json
import logging
import gc
from typing import Any, TYPE_CHECKING

from openpi.policies import policy as _policy

# Import B1K-specific policy_config
from b1k.policies import policy_config as _policy_config

if TYPE_CHECKING:
    from b1k.training import config as _config


class CheckpointSwitcher:
    """Loads different checkpoints per task. Only one checkpoint in memory at a time.
    All 50 tasks must be mapped in task_checkpoint_mapping.json.
    """
    
    def __init__(
        self,
        config_path: str,
        training_config: "TrainConfig",
        sample_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the checkpoint switcher.
        
        Args:
            config_path: Path to task_checkpoint_mapping.json (REQUIRED)
            training_config: Training config for loading policies
            sample_kwargs: kwargs for policy sampling (e.g., num_steps)
        """
        if not config_path:
            raise ValueError("config_path is required for checkpoint switching")
        
        self.config_path = config_path
        self.training_config = training_config
        self.sample_kwargs = sample_kwargs or {}
        
        # Task ID → checkpoint name mapping (built from config)
        self.task_to_checkpoint: dict[int, str] = {}
        # Checkpoint name → path mapping
        self.checkpoint_paths: dict[str, str] = {}
        
        # Currently loaded checkpoint
        self.current_policy: _policy.Policy | None = None
        self.current_checkpoint_name: str | None = None
        
        # Load and validate mapping
        self._load_mapping()
        self._validate_all_tasks_covered()
        
        logging.info(f"Checkpoint switcher initialized with {len(self.checkpoint_paths)} checkpoints")
        logging.info(f"All 50 tasks explicitly mapped to checkpoints")
    
    def _load_mapping(self):
        """Load and validate the checkpoint→tasks mapping."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"Checkpoint mapping file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in checkpoint mapping file: {e}")
            raise
        
        if "checkpoints" not in config:
            raise ValueError("Checkpoint mapping file must contain 'checkpoints' key")
        
        # Build task→checkpoint and checkpoint→path mappings
        seen_tasks = set()
        for checkpoint_name, checkpoint_info in config["checkpoints"].items():
            if "path" not in checkpoint_info:
                raise ValueError(f"Checkpoint '{checkpoint_name}' missing 'path' field")
            if "tasks" not in checkpoint_info:
                raise ValueError(f"Checkpoint '{checkpoint_name}' missing 'tasks' field")
            
            checkpoint_path = checkpoint_info["path"]
            tasks = checkpoint_info["tasks"]
            
            # Validate tasks are unique
            for task_id in tasks:
                if task_id in seen_tasks:
                    raise ValueError(f"Task {task_id} is assigned to multiple checkpoints")
                seen_tasks.add(task_id)
            
            self.checkpoint_paths[checkpoint_name] = checkpoint_path
            for task_id in tasks:
                self.task_to_checkpoint[task_id] = checkpoint_name
            
            logging.info(f"Checkpoint '{checkpoint_name}' -> {len(tasks)} tasks: {sorted(tasks)}")
    
    def _validate_all_tasks_covered(self):
        """Validate that all 50 tasks are explicitly mapped to checkpoints."""
        all_tasks = set(range(50))
        mapped_tasks = set(self.task_to_checkpoint.keys())
        missing_tasks = sorted(all_tasks - mapped_tasks)
        
        if missing_tasks:
            raise ValueError(
                f"All 50 tasks must be explicitly mapped to checkpoints. "
                f"Missing tasks: {missing_tasks}"
            )
    
    def get_checkpoint_for_task(self, task_id: int) -> str:
        """Get which checkpoint should handle this task.
        
        Args:
            task_id: Task ID (0-49)
            
        Returns:
            Checkpoint name
            
        Raises:
            ValueError: If task_id not mapped (should never happen after validation)
        """
        if task_id not in self.task_to_checkpoint:
            raise ValueError(f"Task {task_id} not mapped to any checkpoint")
        
        return self.task_to_checkpoint[task_id]
    
    def get_policy_for_task(self, task_id: int) -> _policy.Policy:
        """Get policy for task_id, loading new checkpoint if needed.
        
        Args:
            task_id: Task ID (0-49)
            
        Returns:
            Policy for the requested task
        """
        target_checkpoint = self.get_checkpoint_for_task(task_id)
        
        # If already loaded, return it
        if self.current_checkpoint_name == target_checkpoint and self.current_policy is not None:
            logging.debug(f"Task {task_id} using already-loaded checkpoint '{target_checkpoint}'")
            return self.current_policy
        
        # Need to switch checkpoints
        logging.info(f"Task {task_id} requires checkpoint '{target_checkpoint}'")
        
        # Unload current policy and free JAX/GPU memory
        if self.current_policy is not None:
            logging.info(f"Unloading checkpoint '{self.current_checkpoint_name}'")
            del self.current_policy
            self.current_policy = None
            
            # Force garbage collection for JAX models
            gc.collect()
            
            # Clear JAX compilation cache and device memory
            try:
                import jax
                jax.clear_caches()
                logging.info("Cleared JAX caches and device memory")
            except Exception as e:
                logging.warning(f"Could not clear JAX caches: {e}")
        
        # Load new checkpoint
        checkpoint_path = self.checkpoint_paths[target_checkpoint]
        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            self.current_policy = _policy_config.create_trained_policy(
                self.training_config,
                checkpoint_path,
                sample_kwargs=self.sample_kwargs
            )
            self.current_checkpoint_name = target_checkpoint
            logging.info(f"Successfully loaded checkpoint '{target_checkpoint}'")
        except Exception as e:
            logging.error(f"Failed to load checkpoint '{target_checkpoint}': {e}")
            raise
        
        return self.current_policy

