"""Checkpoint management with FAST tokenizer saving.

Reference: https://github.com/Physical-Intelligence/openpi
"""

from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import shutil
from typing import Protocol

from etils import epath
import flax.traverse_util
import jax
import flax.nnx as nnx

import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils

# Use our custom normalize (has per_timestamp fields and can save/load them)
from b1k.shared import normalize as _normalize


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)
        
        # Save FAST tokenizer if it exists
        model = nnx.merge(state.model_def, state.params)
        # Check if model has FAST enabled (indicated by having fast_token_embedding)
        if hasattr(model, 'fast_token_embedding'):
            # Get the checkpoint directory to determine assets base dir
            # Directory structure: checkpoint_dir/step/assets/asset_id/fast_tokenizer
            # We need to copy from: assets_base_dir/asset_id/fast_tokenizer
            # The assets_base_dir is stored in the config, but we can infer it from checkpoint_dir
            checkpoint_dir = checkpoint_manager.directory
            
            # Infer assets_base_dir from checkpoint structure
            # checkpoint_dir is like: ./outputs/checkpoints/config_name/exp_name
            # assets_base_dir is like: ./outputs/assets/config_name
            parts = checkpoint_dir.parts
            if 'checkpoints' in parts:
                idx = parts.index('checkpoints')
                assets_base_parts = parts[:idx] + ('assets',) + (parts[idx + 1],)  # Keep config_name
                assets_base_dir = epath.Path(*assets_base_parts)
            else:
                # Fallback: try relative to checkpoint dir
                assets_base_dir = checkpoint_dir.parent.parent / 'assets' / checkpoint_dir.parent.name
            
            fast_tokenizer_source = assets_base_dir / data_config.asset_id / "fast_tokenizer"
            fast_tokenizer_dest = directory / data_config.asset_id / "fast_tokenizer"
            
            if fast_tokenizer_source.exists():
                # Create parent directory
                fast_tokenizer_dest.parent.mkdir(parents=True, exist_ok=True)
                # Copy tokenizer directory
                shutil.copytree(fast_tokenizer_source, fast_tokenizer_dest, dirs_exist_ok=True)
                logging.info(f"Saved FAST tokenizer to checkpoint: {fast_tokenizer_dest}")

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
                
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`.
    # If the restored train_state has an empty `params` dict, it means we're in the non-EMA case
    # and we should restore the main training params.
    if not train_state.params:
        # Non-EMA case: The saved 'params' are the training weights.
        return dataclasses.replace(train_state, params=params["params"])
    else:
        # EMA case: The saved 'params' are the ema_params, and train_state already has the training params.
        return dataclasses.replace(train_state, ema_params=params["params"])
