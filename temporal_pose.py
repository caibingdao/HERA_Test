"""Transition-aligned pose semantics; coordinates use x=column, y=row.

Positive atan2 angles turn from +x towards +y (clockwise on imshow's
downward-y display). Local +x is forward; local +y follows this same basis.
The waypoint simulator has no orientation dynamics: heading is the last
nonzero *executed* displacement direction, initially zero.
"""
import math
import warnings

import numpy as np
import torch


VERSIONS = {"absolute": "absolute_pose_v1", "delta_local": "egocentric_delta_pose_v2"}
INITIAL_HEADING = 0.0
MOTION_EPS = 1e-8


def pose_config(mode="absolute", delta_position_scale=4.0):
    if mode not in VERSIONS:
        raise ValueError(f"unknown temporal pose encoding: {mode!r}")
    scale = float(delta_position_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("delta position scale must be positive and finite")
    return {
        "temporal_pose_encoding": mode,
        "temporal_token_version": VERSIONS[mode],
        "temporal_delta_position_scale": scale,
    }


def checkpoint_pose_config(checkpoint):
    keys = tuple(pose_config())
    present = [key in checkpoint for key in keys]
    if not any(present):
        return pose_config()  # Unversioned legacy checkpoints are absolute only.
    if not all(present):
        raise ValueError("incomplete checkpoint temporal pose metadata")
    config = pose_config(checkpoint[keys[0]], checkpoint[keys[2]])
    if checkpoint[keys[1]] != config[keys[1]]:
        raise ValueError("checkpoint temporal token version contradicts pose encoding")
    return config


def validate_checkpoint_pose(checkpoint, mode, scale, allow_override=False):
    if not isinstance(allow_override, bool):
        raise TypeError("allow pose override must be boolean")
    stored = checkpoint_pose_config(checkpoint)
    expected = pose_config(mode, scale)
    mismatch = stored["temporal_token_version"] != expected["temporal_token_version"]
    if mode == "delta_local":
        mismatch |= stored["temporal_delta_position_scale"] != expected["temporal_delta_position_scale"]
    if mismatch:
        message = f"Temporal token semantics mismatch: checkpoint={stored}, requested={expected}"
        if not allow_override:
            raise RuntimeError(message)
        warnings.warn(message + "; explicit override enabled", RuntimeWarning)
    return expected


def execute_pose_transition(environment, waypoint, heading):
    """Wrap the actual Env.step; never derive post-pose from a proposal."""
    pre = np.array([*environment.robot_location, heading], dtype=np.float64)
    if not np.isfinite(pre).all() or not np.isfinite(waypoint).all():
        raise ValueError("non-finite pre-pose or waypoint")
    reward = environment.step(waypoint)
    position = np.asarray(environment.robot_location, dtype=np.float64).copy()
    if not np.isfinite(position).all() or not math.isfinite(float(reward)):
        raise ValueError("non-finite executed position or reward")
    movement = position - pre[:2]
    post_heading = float(heading)
    if np.linalg.norm(movement) > MOTION_EPS:
        post_heading = float(np.arctan2(movement[1], movement[0]))
    post = np.array([*position, post_heading], dtype=np.float64)
    return reward, torch.tensor(pre, dtype=torch.float32), torch.tensor(post, dtype=torch.float32)


def local_motion_delta(pre_pose, post_pose):
    if post_pose is None:
        raise ValueError("delta_local requires same-transition post_pose; legacy pre-pose alone is insufficient")
    if pre_pose.shape != post_pose.shape or pre_pose.shape[-1] != 3:
        raise ValueError("pre_pose and post_pose must have matching [..., 3] shapes")
    pre, post = pre_pose.float(), post_pose.float()
    if not torch.isfinite(pre).all() or not torch.isfinite(post).all():
        raise ValueError("non-finite transition pose")
    dx, dy = (post[..., :2] - pre[..., :2]).unbind(-1)
    theta = pre[..., 2]
    c, s = theta.cos(), theta.sin()
    turn = post[..., 2] - theta
    return torch.stack((c * dx + s * dy, -s * dx + c * dy,
                        torch.atan2(turn.sin(), turn.cos())), dim=-1)
