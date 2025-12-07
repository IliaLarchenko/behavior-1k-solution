"""B1K Policy Transforms

Transforms BEHAVIOR-1K observations to model format.

Reference: https://github.com/wensi-ai/openpi/blob/behavior/src/openpi/policies/b1k_policy.py
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES


def make_b1k_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/egocentric_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(23),
        "prompt": "do something",
    }

def extract_state_from_proprio(proprio_data):
    """
    We assume perfect correlation for the two gripper fingers.
    """
    # extract joint position
    base_qvel = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]]  # 3
    trunk_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["trunk_qpos"]]  # 4
    arm_left_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_left_qpos"]]  #  7
    arm_right_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_right_qpos"]]  #  7
    
    # Extract raw gripper widths and normalize to [-1, 1] to match action space
    left_gripper_raw = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_left_qpos"]].sum(axis=-1, keepdims=True)
    right_gripper_raw = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_right_qpos"]].sum(axis=-1, keepdims=True)
    
    # Normalize gripper widths from [0, 0.1] to [-1, 1] 
    # Based on statistics: physical range is [0, 0.1], action range is [-1, 1]
    # Formula: normalized = 2 * (raw / max_width) - 1
    MAX_GRIPPER_WIDTH = 0.1  # From statistics q99 values
    left_gripper_width = 2.0 * (left_gripper_raw / MAX_GRIPPER_WIDTH) - 1.0
    right_gripper_width = 2.0 * (right_gripper_raw / MAX_GRIPPER_WIDTH) - 1.0

    # Original baseline uses incorrect order for the state
    return np.concatenate([
        base_qvel,
        trunk_qpos,
        arm_left_qpos,
        left_gripper_width,    # Now normalized [-1, 1]
        arm_right_qpos,
        right_gripper_width,   # Now normalized [-1, 1]
    ], axis=-1)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class B1kInputs(transforms.DataTransformFn):
    # Determines which model will be used (not actually used in B1K, kept for compatibility)
    model_type: _model.ModelType | str = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:

        proprio_data = data["observation/state"]
        # extract joint position
        state = extract_state_from_proprio(proprio_data)
        if "actions" in data:
            action =  data["actions"]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/egocentric_camera"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])

        # For B1K, always use 3 cameras (base, left_wrist, right_wrist)
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image_left, wrist_image_right)
        image_masks = (np.True_, np.True_, np.True_)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
            
        # Preserve task_index for PI_BEHAVIOR model
        if "task_index" in data:
            inputs["task_index"] = data["task_index"]
            
        # Preserve tokenized_prompt for PI_BEHAVIOR model
        if "tokenized_prompt" in data:
            inputs["tokenized_prompt"] = data["tokenized_prompt"]
        if "tokenized_prompt_mask" in data:
            inputs["tokenized_prompt_mask"] = data["tokenized_prompt_mask"]
            
        # Preserve subtask_state for PI_BEHAVIOR model
        if "subtask_state" in data:
            inputs["subtask_state"] = data["subtask_state"]
            
        # Preserve timestamp and episode_index for subtask state computation
        if "timestamp" in data:
            inputs["timestamp"] = data["timestamp"]
        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]
            
        # Preserve initial_actions for inpainting
        if "initial_actions" in data:
            initial_actions = data["initial_actions"]
            # Pad initial_actions from 23 dimensions to 32 dimensions (model's action_dim)
            if initial_actions.shape[-1] < 32:
                padding_dim = 32 - initial_actions.shape[-1]
                padding = np.zeros(initial_actions.shape[:-1] + (padding_dim,))
                initial_actions = np.concatenate([initial_actions, padding], axis=-1)
            inputs["initial_actions"] = initial_actions

        return inputs


@dataclasses.dataclass(frozen=True)
class B1kOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Return actions (truncated to 23 dims) and preserve subtask predictions
        result = {"actions": np.asarray(data["actions"][:, :23])}
        
        # Preserve subtask prediction fields for PI_BEHAVIOR models
        if "subtask_logits" in data:
            result["subtask_logits"] = data["subtask_logits"]
        if "predicted_stage" in data:
            result["predicted_stage"] = data["predicted_stage"]
            
        return result
