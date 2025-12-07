"""Data transforms for BEHAVIOR-1K dataset.

Standard transforms imported from OpenPI.
B1K-specific: TaskIndexToTaskId, ComputeSubtaskStateFromMeta, TokenizeFASTActions

Reference: https://github.com/wensi-ai/openpi/tree/behavior
"""

import dataclasses
import logging
import numpy as np

# Import all standard transforms from OpenPI
from openpi.transforms import (
    DataTransformFn,
    DataDict,
    Group,
    CompositeTransform,
    compose,
    RepackTransform,
    Normalize,
    Unnormalize,
    ResizeImages,
    SubsampleActions,
    DeltaActions,
    AbsoluteActions,
    PadStatesAndActions,
    PromptFromLeRobotTask,  # Not used for PI_BEHAVIOR but kept for compatibility
    flatten_dict,
    unflatten_dict,
    transform_dict,
    apply_tree,
    pad_to_dim,
    make_bool_mask,
)

from b1k.models.pi_behavior_config import TASK_NUM_STAGES
from b1k.shared.normalize import NormStats


@dataclasses.dataclass(frozen=True)
class TaskIndexToTaskId(DataTransformFn):
    """Directly converts task_index to task_id for PI_BEHAVIOR model.
    
    PI_BEHAVIOR uses task embeddings instead of text prompts. This transform
    converts the dataset's task_index to task_id and prepares tokenized_prompt
    as [task_id, subtask_state] for the model.
    
    Assumes:
    - data["task_index"] exists (from dataset)
    - data["subtask_state"] exists (computed by ComputeSubtaskStateFromMeta)
    
    Creates:
    - data["tokenized_prompt"]: np.array([task_id, subtask_state], dtype=int32)
    - data["tokenized_prompt_mask"]: np.array([True, True], dtype=bool)
    """
    
    # Optional task remapping (dataset task_index → model task_id)
    # If None, assumes direct mapping (task_index == task_id)
    task_mapping: dict[int, int] | None = None
    
    def __call__(self, data: DataDict) -> DataDict:
        # During inference, task_id might be provided directly instead of task_index
        if "task_id" in data:
            # Direct task_id provided (inference mode)
            task_id = int(data["task_id"])
        elif "task_index" in data:
            # task_index provided (training mode)
            task_index = int(data["task_index"])
            
            # Apply mapping if provided, otherwise use task_index directly as task_id
            if self.task_mapping is not None:
                if task_index not in self.task_mapping:
                    raise ValueError(f"task_index {task_index} not found in mapping")
                task_id = self.task_mapping[task_index]
            else:
                task_id = task_index
        else:
            # During inference, if neither is provided, this transform should be skipped
            # The tokenized_prompt should already be set up correctly
            if "tokenized_prompt" in data:
                return data  # Already has tokenized_prompt, skip this transform
            raise ValueError("Either task_index, task_id, or tokenized_prompt is required for PI_BEHAVIOR model")

        # Pack task_id and subtask_state (if available) into tokenized_prompt
        if "subtask_state" in data:
            subtask_state = int(data["subtask_state"])
            prompt_tokens = np.array([task_id, subtask_state], dtype=np.int32)  # [task_id, subtask_state]
            prompt_mask = np.array([True, True], dtype=bool)
        else:
            prompt_tokens = np.array([task_id], dtype=np.int32)  # Just [task_id]
            prompt_mask = np.array([True], dtype=bool)

        return {
            **data, 
            "tokenized_prompt": prompt_tokens,
            "tokenized_prompt_mask": prompt_mask
        }


@dataclasses.dataclass(frozen=True)
class ComputeSubtaskStateFromMeta(DataTransformFn):
    """Computes subtask state from timestamp using dataset.meta.episodes.
    
    Divides episode into task-specific number of stages based on episode length.
    Requires dataset reference to access episode metadata.
    
    Args:
        dataset: Dataset instance with meta.episodes containing episode_length
    
    Assumes:
    - data["episode_index"] exists
    - data["timestamp"] exists (frame index within episode)
    - data["task_index"] exists (for task-specific stage count)
    - dataset.meta.episodes contains episode_length for each episode
    
    Creates:
    - data["subtask_state"]: Stage index (0 to num_stages-1)
    """
    
    dataset: object | None = None  # Will be set by data loader
    
    def __call__(self, data: DataDict) -> DataDict:
        if self.dataset is None:
            # During inference or when dataset is not available, use default
            data["subtask_state"] = np.array(0, dtype=np.int32)
            return data
        
        if "episode_index" not in data or "timestamp" not in data or "task_index" not in data:
            # Missing required fields, default to stage 0
            data["subtask_state"] = np.array(0, dtype=np.int32)
            return data
        
        episode_index = int(data["episode_index"])
        timestamp = float(data["timestamp"])
        task_index = int(data["task_index"])
        
        # Validate task_index
        if not (0 <= task_index < 50):
            logging.warning(f"Invalid task_index {task_index}, using stage 0")
            data["subtask_state"] = np.array(0, dtype=np.int32)
            return data
        
        # Get number of stages for this task
        num_stages = TASK_NUM_STAGES[task_index]
        
        # Get episode length from dataset metadata
        if not hasattr(self.dataset, 'meta') or not hasattr(self.dataset.meta, 'episodes'):
            # No metadata available
            data["subtask_state"] = np.array(0, dtype=np.int32)
            return data
            
        meta_episodes = self.dataset.meta.episodes
        
        if episode_index not in meta_episodes:
            logging.warning(f"Episode {episode_index} not found in metadata, using stage 0")
            data["subtask_state"] = np.array(0, dtype=np.int32)
            return data
            
        episode_info = meta_episodes[episode_index]
        
        # Try 'length' first (standard key), then 'episode_length' (alternative)
        episode_length = episode_info.get('length', episode_info.get('episode_length', None))
        
        if episode_length is None or episode_length <= 0:
            logging.warning(f"Invalid episode_length for episode {episode_index}, using stage 0")
            data["subtask_state"] = np.array(0, dtype=np.int32)
            return data
        
        episode_length = float(episode_length)
        
        # CRITICAL: Convert timestamp (in seconds) to frames (30 FPS)
        # Dataset provides timestamp in seconds, episode_length is in frames
        current_step = timestamp * 30.0
        
        # Divide episode into num_stages equal parts
        frames_per_stage = episode_length / num_stages
        
        # Compute current stage (0-indexed)
        subtask_state = int(current_step / frames_per_stage)
        
        # Clamp to valid range [0, num_stages-1]
        subtask_state = max(0, min(subtask_state, num_stages - 1))
        
        data["subtask_state"] = np.array(subtask_state, dtype=np.int32)
        return data


@dataclasses.dataclass(frozen=True)
class TokenizeFASTActions(DataTransformFn):
    """Tokenize actions to FAST discrete tokens for PI_BEHAVIOR auxiliary training.
    
    Converts continuous actions to discrete tokens using a trained FAST tokenizer.
    This is used for auxiliary training to improve action quality.
    
    Note: Uses GLOBAL normalization even if per-timestamp normalization is enabled
    elsewhere, to preserve temporal smoothness for better DCT compression.
    
    Args:
        tokenizer_path: Path to trained FAST tokenizer directory
        encoded_dim_ranges: List of (start, end) tuples for dimensions to encode
        max_fast_tokens: Maximum number of tokens (truncate/pad)
        norm_stats: Normalization stats for denorm/renorm if per-timestamp is used
        use_per_timestamp: Whether per-timestamp normalization is being used
    
    Assumes:
    - data["actions"] exists and is normalized
    - data["state"] exists (for delta computation)
    
    Creates:
    - data["fast_tokens"]: Discrete token indices [max_fast_tokens]
    - data["fast_token_mask"]: Boolean mask for valid tokens [max_fast_tokens]
    """
    
    # Path to FAST tokenizer
    tokenizer_path: str
    # Action dimensions to encode: [(start, end), ...]
    encoded_dim_ranges: list[tuple[int, int]]
    # Max tokens to predict
    max_fast_tokens: int = 180
    # Normalization stats (for denorm/renorm if per-timestamp is used)
    norm_stats: dict[str, NormStats] | None = None
    # Whether per-timestamp normalization is being used
    use_per_timestamp: bool = False
    
    def _get_tokenizer(self):
        """Lazy load tokenizer (called in each worker process)."""
        # Check if already loaded in this process
        if not hasattr(self, '_tokenizer'):
            import importlib.util
            from pathlib import Path
            
            tokenizer_dir = Path(self.tokenizer_path)
            processor_file = tokenizer_dir / "processing_action_tokenizer.py"
            
            if not processor_file.exists():
                raise FileNotFoundError(f"Tokenizer processing file not found at {processor_file}")
            
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("processing_action_tokenizer", processor_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the UniversalActionProcessor class
            UniversalActionProcessor = module.UniversalActionProcessor
            
            # Load the tokenizer using from_pretrained
            tokenizer = UniversalActionProcessor.from_pretrained(str(tokenizer_dir))
            
            # Store in this object (won't be pickled due to lazy loading pattern)
            object.__setattr__(self, '_tokenizer', tokenizer)
        
        return self._tokenizer
    
    def __call__(self, data: DataDict) -> DataDict:
        # Only tokenize if actions are present (training time)
        if "actions" not in data:
            return data
        
        actions = data["actions"]  # [H, D] - may be per-timestamp normalized
        
        # Extract encoded dimensions
        encoded_chunks = []
        for start, end in self.encoded_dim_ranges:
            encoded_chunks.append(actions[:, start:end])
        encoded_actions = np.concatenate(encoded_chunks, axis=-1)  # [H, D_encoded]
        
        # Convert to global QUANTILE normalization for FAST (always, regardless of per_timestamp)
        # This clips outliers and provides consistent [-1, 1] range for DCT compression
        if self.norm_stats is not None:
            action_stats = self.norm_stats['actions']
            
            # Extract stats for encoded dimensions only
            # Build dimension indices from ranges
            encoded_dims = []
            for start, end in self.encoded_dim_ranges:
                encoded_dims.extend(range(start, end))
            encoded_dims = np.array(encoded_dims)
            
            # Denormalize to get back to raw delta actions
            if self.use_per_timestamp:
                # Denormalize from per-timestamp MEAN/STD normalization
                per_ts_mean = action_stats.per_timestamp_mean[:, encoded_dims]  # [H, D_encoded]
                per_ts_std = action_stats.per_timestamp_std[:, encoded_dims]    # [H, D_encoded]
                actions_delta = encoded_actions * per_ts_std + per_ts_mean
            else:
                # Denormalize from global MEAN/STD normalization
                global_mean = action_stats.mean[encoded_dims]  # [D_encoded]
                global_std = action_stats.std[encoded_dims]    # [D_encoded]
                actions_delta = encoded_actions * global_std + global_mean
            
            # Renormalize using GLOBAL QUANTILE normalization (for FAST)
            # This clips to [q01, q99] and maps to [-1, 1]
            q01 = action_stats.q01[encoded_dims]  # [D_encoded]
            q99 = action_stats.q99[encoded_dims]  # [D_encoded]
            actions_delta = np.clip(actions_delta, q01, q99)
            encoded_actions = 2.0 * (actions_delta - q01) / np.maximum(q99 - q01, 1e-6) - 1.0
        
        # Tokenize
        tokenizer = self._get_tokenizer()
        tokens = tokenizer(encoded_actions[None])[0]
        
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        
        # Truncate or pad to max_fast_tokens
        if len(tokens) > self.max_fast_tokens:
            tokens = tokens[:self.max_fast_tokens]
            mask = np.ones(self.max_fast_tokens, dtype=bool)
        else:
            mask = np.concatenate([
                np.ones(len(tokens), dtype=bool),
                np.zeros(self.max_fast_tokens - len(tokens), dtype=bool)
            ])
            tokens = np.pad(tokens, (0, self.max_fast_tokens - len(tokens)), constant_values=0)
        
        data["fast_tokens"] = tokens.astype(np.int32)
        data["fast_token_mask"] = mask
        
        return data

