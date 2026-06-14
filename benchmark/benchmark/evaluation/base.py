"""Base evaluator interfaces and shared trajectory preprocessing.

Defines:
  - BaseEvaluator: abstract interface (evaluate / save_visualization) shared
    by all four evaluator types.
  - BaseTrajectoryEvaluator: common trajectory loading, NaN filtering, and
    evo trajectory construction for TrajectoryEvaluator and AUCEvaluator.
"""

import abc
import logging
from typing import Any, Dict, Optional

import numpy as np

from benchmark.core.trajectory_utils import (
    filter_valid_pose_pairs,
    array_to_evo_trajectory,
)


class BaseEvaluator(abc.ABC):
    """Top-level evaluator interface.

    All concrete evaluators (trajectory, AUC, depth, point cloud) share the
    ``evaluate(gt_loader, pred_loader, logger)`` signature so that the
    orchestrator in ``benchmark.core.evaluator`` can dispatch them uniformly.
    """

    @abc.abstractmethod
    def evaluate(
        self,
        gt_loader,
        pred_loader,
        logger: Optional[logging.Logger] = None,
    ) -> Any:
        """Run evaluation and return metrics (dict, tuple, etc.)."""


class BaseTrajectoryEvaluator(BaseEvaluator):
    """Common base for trajectory-based evaluators.

    Handles the shared preamble of:
      1. Loading GT / predicted trajectories from loaders.
      2. Selecting matching GT poses via ``frame_index_map``.
      3. Filtering NaN pose pairs.
      4. Constructing evo ``PoseTrajectory3D`` objects.

    Subclasses override ``_compute_metrics`` (and optionally
    ``save_visualization``) to implement their specific metric logic.
    """

    def _ensure_logger(self, logger: Optional[logging.Logger]) -> logging.Logger:
        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            logger = logging.getLogger(__name__)
        return logger

    def _load_and_prepare_trajectories(
        self,
        gt_loader,
        pred_loader,
        logger: logging.Logger,
    ):
        """Load trajectories, filter NaN poses, build evo objects.

        Returns:
            Tuple of (traj_ref, traj_est, gt_poses, pred_poses, timestamps_float,
                      pred_frame_indices)
            where traj_ref / traj_est are evo PoseTrajectory3D objects and the
            raw numpy arrays are also returned for subclass use.
        """
        gt_traj = gt_loader.load_trajectory()
        pred_traj = pred_loader.load_trajectory()

        if gt_traj is None:
            raise FileNotFoundError(
                f"GT trajectory not found: {gt_loader.artifact.traj_file}"
            )
        if pred_traj is None:
            raise FileNotFoundError(
                f"Predicted trajectory not found: {pred_loader.artifact.traj_file}"
            )

        pred_frame_indices = pred_loader.get_frame_indices()
        gt_poses = gt_traj[pred_frame_indices]  # (K, 4, 4)
        pred_poses = pred_traj                  # (K, 4, 4), always dense/valid

        if len(gt_poses) == 0 or len(pred_poses) == 0:
            raise ValueError("Empty trajectory — no frames to evaluate")

        timestamps_float = np.array(pred_frame_indices, dtype=float)

        gt_poses, pred_poses, timestamps_float = filter_valid_pose_pairs(
            gt_poses, pred_poses, timestamps_float, logger
        )

        traj_ref = array_to_evo_trajectory(gt_poses, timestamps_float)
        traj_est = array_to_evo_trajectory(pred_poses, timestamps_float)

        return (
            traj_ref, traj_est,
            gt_poses, pred_poses, timestamps_float,
            pred_frame_indices,
        )

    def evaluate(
        self,
        gt_loader,
        pred_loader,
        logger: Optional[logging.Logger] = None,
    ) -> Any:
        logger = self._ensure_logger(logger)
        prepared = self._load_and_prepare_trajectories(gt_loader, pred_loader, logger)
        return self._compute_metrics(prepared, gt_loader, pred_loader, logger)

    @abc.abstractmethod
    def _compute_metrics(
        self,
        prepared,
        gt_loader,
        pred_loader,
        logger: logging.Logger,
    ) -> Any:
        """Compute metrics from already-prepared trajectory data.

        Args:
            prepared: Tuple returned by ``_load_and_prepare_trajectories``.
            gt_loader: Ground truth loader (for extra data access if needed).
            pred_loader: Prediction loader.
            logger: Logger instance.
        """
