"""Shared trajectory utility functions.

Centralized pose filtering, SE3 orthogonalization, and evo trajectory conversion
used by trajectory and AUC evaluators.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from evo.core.trajectory import PoseTrajectory3D


def filter_valid_pose_pairs(
    gt_poses: np.ndarray,
    pred_poses: np.ndarray,
    timestamps: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop frame pairs where GT or pred pose contains NaN/Inf.

    Logs a WARNING if any pairs are dropped (indicates some GT frames lack
    valid poses, e.g. TUM RGB-D timestamp association gaps).

    Returns:
        Filtered (gt_poses, pred_poses, timestamps), all with NaN-free poses.
    Raises:
        ValueError: If no valid pairs remain after filtering.
    """
    gt_valid   = np.isfinite(gt_poses.reshape(len(gt_poses), -1)).all(axis=1)
    pred_valid = np.isfinite(pred_poses.reshape(len(pred_poses), -1)).all(axis=1)
    mask = gt_valid & pred_valid
    n_dropped = int((~mask).sum())
    if n_dropped > 0:
        msg = (f"Dropped {n_dropped}/{len(mask)} frame(s) with NaN GT/pred pose "
               "(GT trajectory is incomplete for these frames)")
        if logger:
            logger.warning(msg)
        else:
            print(f"WARNING: {msg}")
    gt_poses   = gt_poses[mask]
    pred_poses = pred_poses[mask]
    timestamps = timestamps[mask]
    if len(gt_poses) == 0:
        raise ValueError("No valid GT/pred pose pairs after NaN filtering")
    return gt_poses, pred_poses, timestamps


def orthogonalize_se3(pose: np.ndarray) -> np.ndarray:
    """Orthogonalize the rotation part of a 4x4 SE3 matrix via SVD.

    Required because evo validates SO(3) membership, but some datasets
    (e.g., 7Scenes KinectFusion) have slightly non-orthogonal R.

    Args:
        pose: 4x4 transformation matrix

    Returns:
        4x4 matrix with orthogonalized rotation
    """
    result = pose.copy()
    R = result[:3, :3]
    if not np.all(np.isfinite(R)):
        raise ValueError(f"Rotation matrix contains non-finite values (NaN/Inf): {R}")
    try:
        U, _, Vh = np.linalg.svd(R)
    except np.linalg.LinAlgError:
        # Fall back to QR decomposition if SVD fails to converge
        Q, _ = np.linalg.qr(R)
        result[:3, :3] = Q
        return result
    R_ortho = U @ Vh
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vh
    result[:3, :3] = R_ortho
    return result


def array_to_evo_trajectory(poses: np.ndarray, timestamps: np.ndarray) -> PoseTrajectory3D:
    """Convert valid pose array to evo PoseTrajectory3D.

    Orthogonalizes rotations at the evo boundary to satisfy SO(3) validation.

    Args:
        poses:      (M, 4, 4) array of valid (non-NaN) poses
        timestamps: (M,) float array of timestamps (frame indices as floats)

    Returns:
        evo PoseTrajectory3D object
    """
    poses_se3 = [orthogonalize_se3(poses[i]) for i in range(len(poses))]
    return PoseTrajectory3D(poses_se3=poses_se3, timestamps=timestamps)


# Backward-compatible aliases (legacy underscore-prefixed names)
_filter_valid_pose_pairs = filter_valid_pose_pairs
_orthogonalize_se3 = orthogonalize_se3
_array_to_evo_trajectory = array_to_evo_trajectory
