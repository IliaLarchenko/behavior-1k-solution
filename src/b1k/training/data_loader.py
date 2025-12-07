"""Data loading for BEHAVIOR-1K dataset.

Reference: https://github.com/wensi-ai/openpi/tree/behavior
"""

import logging
import time

# Import all base data loading from OpenPI
from openpi.training.data_loader import (
    Dataset,
    IterableDataset,
    DataLoader,
    TransformedDataset,
    IterableTransformedDataset,
    FakeDataset,
    TorchDataLoader,
    RLDSDataLoader,
    create_torch_dataset,
    create_rlds_dataset,
    transform_iterable_dataset,
    create_data_loader,
    create_torch_data_loader,
    create_rlds_data_loader,
)

import openpi.training.config as _config
import openpi.transforms as _transforms

from b1k.models.observation import Observation
from b1k.transforms_normalize import NormalizeWithPerTimestamp


class DataLoaderImpl(DataLoader):
    """Custom DataLoader using our Observation with fast_tokens."""
    
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield Observation.from_dict(batch), batch["actions"]


def create_behavior_dataset(data_config: _config.DataConfig, action_horizon: int, seed: int | None = None) -> Dataset:
    """Create a BEHAVIOR-1K dataset for training.
    
    Uses OmniGibson's BehaviorLeRobotDataset for efficient loading of BEHAVIOR-1K data.
    
    Args:
        data_config: Data configuration
        action_horizon: Action horizon for delta timestamps
        seed: Random seed for shuffling. If None, uses random seed based on current time.
    
    Returns:
        Dataset instance with BEHAVIOR-1K data
    """
    from omnigibson.learning.datas.lerobot_dataset import BehaviorLeRobotDataset
    
    # Use random seed if not provided
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed for BehaviorLeRobotDataset: {seed}")
    tasks = [
    "picking_up_trash", # difficulty: 2
    "putting_away_Halloween_decorations", # difficulty: 3
    "cleaning_up_plates_and_food", # difficulty: 3.5
    "setting_mousetraps", # difficulty: 2
    "hiding_Easter_eggs", # difficulty: 2
    "set_up_a_coffee_station_in_your_kitchen", # difficulty: 3
    "putting_dishes_away_after_cleaning", # difficulty: 3
    "preparing_lunch_box", # difficulty: 3
    "loading_the_car", # difficulty: 3.5
    "carrying_in_groceries", # difficulty: 3.5
    "turning_on_radio", # difficulty: 1
    "picking_up_toys", # difficulty: 3.5
    "can_meat", # difficulty: 3.5
    "rearranging_kitchen_furniture", # difficulty: 3
    "putting_up_Christmas_decorations_inside", # difficulty: 2
    "bringing_in_wood", # difficulty: 1.5
    "moving_boxes_to_storage", # difficulty: 1.5
    "bringing_water", # difficulty: 1.5
    "tidying_bedroom", # difficulty: 2
    "outfit_a_basic_toolbox", # difficulty: 2,
    "sorting_vegetables",
    "collecting_childrens_toys",
    "putting_shoes_on_rack",
    "boxing_books_up_for_storage",
    "storing_food",
    "clearing_food_from_table_into_fridge",
    "assembling_gift_baskets",
    "sorting_household_items",
    "getting_organized_for_work",
    "clean_up_your_desk",
    "setting_the_fire",
    "clean_boxing_gloves",
    "wash_a_baseball_cap",
    "wash_dog_toys",
    "hanging_pictures",
    "attach_a_camera_to_a_tripod",
    "clean_a_patio",
    "clean_a_trumpet",
    "spraying_for_bugs",
    "spraying_fruit_trees",
    "make_microwave_popcorn",
    "cook_cabbage",
    "make_pizza",
    "chop_an_onion",
    "slicing_vegetables",
    "chopping_wood",
    "canning_food",
    "cook_hot_dogs",
    "cook_bacon",
    "freeze_pies",
    ]
    
    dataset = BehaviorLeRobotDataset(
        repo_id=data_config.repo_id,
        root=data_config.behavior_dataset_root,
        tasks=tasks,
        modalities=["rgb"],
        local_only=True,
        delta_timestamps={
            key: [t / 30.0 for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        episodes=data_config.episodes_index,
        chunk_streaming_using_keyframe=False,
        shuffle=True,
        seed=seed,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset.meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform dataset with B1K-specific per-timestamp normalization support.
    
    CRITICAL: This overrides wensi-ai's transform_dataset to pass use_per_timestamp to Normalize.
    wensi-ai's version doesn't support per-timestamp normalization which causes huge action losses!
    """
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # Build transform list
    transforms_list = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        # Use custom Normalize with per-timestamp support (wensi-ai's doesn't have it!)
        NormalizeWithPerTimestamp(
            norm_stats, 
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm  # CRITICAL: Per-timestamp normalization!
        ),
    ]
    
    # Add subtask state computation for PI_BEHAVIOR models (needs dataset reference)
    model_transforms = []
    for transform in data_config.model_transforms.inputs:
        # ComputeSubtaskStateFromMeta needs dataset reference to access episode lengths
        if hasattr(transform, '__class__') and transform.__class__.__name__ == 'ComputeSubtaskStateFromMeta':
            # Replace placeholder with dataset-aware version
            from b1k import transforms as b1k_transforms
            if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'episodes'):
                model_transforms.append(b1k_transforms.ComputeSubtaskStateFromMeta(dataset=dataset))
                logging.info("Added dataset-aware ComputeSubtaskStateFromMeta transform")
            else:
                logging.warning("Skipping subtask state computation - dataset has no meta.episodes")
        else:
            model_transforms.append(transform)
    
    transforms_list.extend(model_transforms)

    return TransformedDataset(dataset, transforms_list)


def extract_episode_lengths_from_dataset(dataset) -> dict[int, float]:
    """Extract episode lengths from B1K dataset metadata.
    
    Args:
        dataset: BehaviorLeRobotDataset instance
        
    Returns:
        Dictionary mapping episode_index to episode_length (in frames)
        
    Raises:
        ValueError: If dataset doesn't have required metadata
    """
    if not hasattr(dataset, 'episode_data_index'):
        raise ValueError("Dataset must have episode_data_index attribute")
    
    episode_data_index = dataset.episode_data_index
    if 'to' not in episode_data_index or 'from' not in episode_data_index:
        raise ValueError("episode_data_index must have 'to' and 'from' keys")
    
    episode_to = episode_data_index['to'] 
    episode_from = episode_data_index['from']
    episodes = dataset.episodes
    
    episode_lengths = {}
    for i, episode_index in enumerate(episodes):
        if i < len(episode_to) and i < len(episode_from):
            episode_length = episode_to[i] - episode_from[i]
            episode_lengths[episode_index] = float(episode_length)
    
    logging.info(f"Extracted {len(episode_lengths)} episode lengths from dataset")
    return episode_lengths


def create_behavior_data_loader(
    config: _config.TrainConfig,
    *,
    sharding=None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader:
    """Create a data loader for BEHAVIOR-1K training."""
    import jax
    import time
    
    data_config = config.data.create(config.assets_dirs, config.model)
    
    # Use random seed if not provided
    seed = config.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed: {seed}")
    
    dataset = create_behavior_dataset(data_config, action_horizon=config.model.action_horizon, seed=seed)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
    )
    
    return DataLoaderImpl(data_config, data_loader)
