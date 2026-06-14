"""Base evaluator interfaces and trajectory evaluator base class.

Defines the common evaluator contract and shared trajectory preprocessing
pipeline used by TrajectoryEvaluator and AUCEvaluator.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from benchmark.core.trajectory_utils import filter_valid_pose_pairs


class BaseEvaluator(ABC):
    """Abstract base interface for all evaluators.

    All evaluators share the ``evaluate(gt_loader, pred_loader, logger)``
    signature so they can be orchestrated uniformly by
    :class:`benchmark.core.evaluator.Evaluator`.
    """

    @abstractmethod
    def evaluate(self, gt_loader, pred_loader,
                 logger: Optional[logging.Logger] = None) -> Dict:
        """Run evaluation and return metrics dictionary."""
        ...


class BaseTrajectoryEvaluator(BaseEvaluator):
    """Base class for evaluators that compare camera trajectories.

    Encapsulates the common preprocessing pipeline shared by
    :class:`TrajectoryEvaluator` (ATE/RPE) and :class:`AUCEvaluator` (pose AUC):

    1. Load GT and predicted trajectories
    2. Map predicted frames to GT via ``frame_index_map``
    3. Filter NaN/Inf pose pairs
    4. Return matched arrays ready for metric computation

    Subclasses implement :meth:`evaluate` with their specific metric logic.
    """

    def __init__(self, align: bool = True, correct_scale: bool = True):
        """Initialize trajectory evaluator base.

        Args:
            align:         Whether to align trajectories before evaluation
            correct_scale: Whether to correct scale during alignment
        """
        self.align = align
        self.correct_scale = correct_scale

    @staticmethod
    def _ensure_logger(logger: Optional[logging.Logger]) -> logging.Logger:
        """Return *logger* or create a default one."""
        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            logger = logging.getLogger(__name__)
        return logger

    def _load_trajectories(self, gt_loader, pred_loader,
                           logger: Optional[logging.Logger] = None):
        """Load and preprocess matched trajectory pairs.

        Common pipeline: load trajectories → frame_index_map → NaN filter.

        Args:
            gt_loader:   BSSLoader for ground truth directory
            pred_loader: BSSLoader for predicted directory
            logger:      Optional logger

        Returns:
            Tuple of ``(gt_poses, pred_poses, timestamps, num_frames)``

        Raises:
            FileNotFoundError: If GT or predicted trajectory is missing
            ValueError:        If no valid pose pairs remain after filtering
        """
        logger = self._ensure_logger(logger)

        gt_traj = gt_loader.load_trajectory()
        pred_traj = pred_loader.load_trajectory()

        if gt_traj is None:
            raise FileNotFoundError(
                f"GT trajectory not found: {gt_loader.artifact.traj_file}")
        if pred_traj is None:
            raise FileNotFoundError(
                f"Predicted trajectory not found: {pred_loader.artifact.traj_file}")

        # Use frame_index_map to select matching GT poses for sparse SLAM outputs.
        # For dense methods the map is identity [0, 1, ..., N-1].
        pred_frame_indices = pred_loader.get_frame_indices()
        gt_poses   = gt_traj[pred_frame_indices]  # (K, 4, 4)
        pred_poses = pred_traj                    # (K, 4, 4), always dense/valid

        if len(gt_poses) == 0 or len(pred_poses) == 0:
            raise ValueError("Empty trajectory — no frames to evaluate")

        timestamps = np.array(pred_frame_indices, dtype=float)
        gt_poses, pred_poses, timestamps = filter_valid_pose_pairs(
            gt_poses, pred_poses, timestamps, logger)

        logger.info(f"Evaluating trajectory on {len(gt_poses)} frame pairs")
        return gt_poses, pred_poses, timestamps, len(gt_poses)

    @abstractmethod
    def evaluate(self, gt_loader, pred_loader,
                 logger: Optional[logging.Logger] = None) -> Dict:
        """Run evaluation and return metrics dictionary."""
        ...
