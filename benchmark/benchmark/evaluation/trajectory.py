"""Trajectory evaluation using evo library (ATE and RPE metrics)."""

import copy
import logging
from pathlib import Path
from typing import Dict, Optional

from evo.core.metrics import PoseRelation, Unit
from evo.tools import plot
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
import matplotlib.pyplot as plt
import numpy as np

from benchmark.core.trajectory_utils import (
    filter_valid_pose_pairs,
    array_to_evo_trajectory,
)
from benchmark.evaluation.base import BaseTrajectoryEvaluator


# Backward-compatible aliases (other modules may still import the underscore names)
_filter_valid_pose_pairs = filter_valid_pose_pairs
_orthogonalize_se3 = None  # moved to trajectory_utils.orthogonalize_se3
_array_to_evo_trajectory = array_to_evo_trajectory


class TrajectoryEvaluator(BaseTrajectoryEvaluator):
    """Evaluates camera trajectories using ATE and RPE metrics."""

    def __init__(self, align: bool = True, correct_scale: bool = True):
        """Initialize trajectory evaluator.

        Args:
            align: Whether to align trajectories before evaluation
            correct_scale: Whether to correct scale during alignment
        """
        self.align = align
        self.correct_scale = correct_scale

    def _compute_metrics(
        self,
        prepared,
        gt_loader,
        pred_loader,
        logger: logging.Logger,
    ) -> Dict[str, float]:
        """Evaluate trajectory using evo library (ATE and RPE).

        Returns:
            Dictionary containing:
            - 'ate': Absolute Trajectory Error (RMSE)
            - 'rpe_trans': RPE translation RMSE
            - 'rpe_rot': RPE rotation RMSE (degrees)
            - 'traj_transform': 4x4 Sim(3) alignment matrix (np.ndarray)
        """
        traj_ref, traj_est, gt_poses, pred_poses, timestamps_float, _ = prepared

        logger.info(f"Evaluating trajectory on {len(gt_poses)} frame pairs")

        # Align estimated trajectory if requested
        traj_est_aligned = copy.deepcopy(traj_est)
        alignment_result = None
        T_align = np.eye(4)
        if self.align:
            alignment_result = traj_est_aligned.align(
                traj_ref, correct_scale=self.correct_scale
            )

        if self.align and alignment_result is not None:
            R, t, scale = alignment_result  # evo returns (rotation, translation, scale)
            T_align = np.eye(4)
            T_align[:3, :3] = scale * R
            T_align[:3, 3] = t.flatten()

        # Compute ATE (Absolute Trajectory Error)
        ape_result = main_ape.ape(
            traj_ref,
            traj_est_aligned,
            est_name='traj',
            pose_relation=PoseRelation.translation_part,
            align=False,
            correct_scale=False,
        )
        ate = ape_result.stats["rmse"]

        # Compute RPE (Relative Pose Error) - rotation
        rpe_rot_result = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            align=self.align,
            correct_scale=self.correct_scale,
            delta=1,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )
        rpe_rot = rpe_rot_result.stats["rmse"]

        # Compute RPE (Relative Pose Error) - translation
        rpe_trans_result = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=self.align,
            correct_scale=self.correct_scale,
            delta=1,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )
        rpe_trans = rpe_trans_result.stats["rmse"]

        return {
            'ate': float(ate),
            'rpe_trans': float(rpe_trans),
            'rpe_rot': float(rpe_rot),
            'traj_transform': T_align,
        }

    def save_visualization(self, gt_loader, pred_loader, output_dir: Path,
                           logger: Optional[logging.Logger] = None) -> None:
        """Visualize aligned trajectories in 4 views (xyz, xy, xz, yz).

        Args:
            gt_loader:   BSSLoader for ground truth directory
            pred_loader: BSSLoader for predicted directory
            output_dir:  Directory to save the visualization (file: trajectory_visualization.png)
            logger:      Optional logger
        """
        logger = self._ensure_logger(logger)

        gt_traj = gt_loader.load_trajectory()
        pred_traj = pred_loader.load_trajectory()

        if gt_traj is None:
            raise FileNotFoundError(f"GT trajectory not found: {gt_loader.artifact.traj_file}")
        if pred_traj is None:
            raise FileNotFoundError(f"Predicted trajectory not found: {pred_loader.artifact.traj_file}")

        pred_frame_indices = pred_loader.get_frame_indices()
        gt_poses = gt_traj[pred_frame_indices]
        pred_poses = pred_traj

        if len(gt_poses) == 0:
            raise ValueError("No valid pose pairs found between reference and estimated trajectories")

        timestamps_float = np.array(pred_frame_indices, dtype=float)

        gt_poses, pred_poses, timestamps_float = filter_valid_pose_pairs(
            gt_poses, pred_poses, timestamps_float, logger)

        traj_ref = array_to_evo_trajectory(gt_poses, timestamps_float)
        traj_est = array_to_evo_trajectory(pred_poses, timestamps_float)

        traj_est_aligned = copy.deepcopy(traj_est)
        if self.align:
            traj_est_aligned.align(traj_ref, correct_scale=self.correct_scale)

        # Build full GT trajectory (all valid poses) for visualization so that a method
        # which only processed a small portion of frames cannot look deceptively good.
        # Alignment above is still computed against the matched subset (traj_ref), which
        # is correct; only the reference drawn in grey uses the full GT here.
        gt_full_valid_mask = np.isfinite(gt_traj.reshape(len(gt_traj), -1)).all(axis=1)
        gt_full_poses = gt_traj[gt_full_valid_mask]
        gt_full_timestamps = np.where(gt_full_valid_mask)[0].astype(float)
        traj_ref_full = array_to_evo_trajectory(gt_full_poses, gt_full_timestamps)

        # Create figure with 4 subplots
        fig = plt.figure(figsize=(12, 12))

        ax = plot.prepare_axis(fig, plot.PlotMode.xyz, subplot_arg=221)
        ax.set_title("XYZ")
        plot.traj(ax, plot.PlotMode.xyz, traj_ref_full, '--', 'gray', label="ref", plot_start_end_markers=True)
        plot.traj(ax, plot.PlotMode.xyz, traj_est_aligned, '-', 'blue', label="est", plot_start_end_markers=True)
        fig.axes.append(ax)

        ax = plot.prepare_axis(fig, plot.PlotMode.xy, subplot_arg=222)
        ax.set_title("XY")
        plot.traj(ax, plot.PlotMode.xy, traj_ref_full, '--', 'gray', label="ref", plot_start_end_markers=True)
        plot.traj(ax, plot.PlotMode.xy, traj_est_aligned, '-', 'blue', label="est", plot_start_end_markers=True)
        fig.axes.append(ax)

        ax = plot.prepare_axis(fig, plot.PlotMode.xz, subplot_arg=223)
        ax.set_title("XZ")
        plot.traj(ax, plot.PlotMode.xz, traj_ref_full, '--', 'gray', label="ref", plot_start_end_markers=True)
        plot.traj(ax, plot.PlotMode.xz, traj_est_aligned, '-', 'blue', label="est", plot_start_end_markers=True)
        fig.axes.append(ax)

        ax = plot.prepare_axis(fig, plot.PlotMode.yz, subplot_arg=224)
        ax.set_title("YZ")
        plot.traj(ax, plot.PlotMode.yz, traj_ref_full, '--', 'gray', label="ref", plot_start_end_markers=True)
        plot.traj(ax, plot.PlotMode.yz, traj_est_aligned, '-', 'blue', label="est", plot_start_end_markers=True)
        fig.axes.append(ax)

        plt.legend()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'trajectory_visualization.png'
        plt.savefig(output_path, dpi=120)
        plt.close(fig)

        logger.info(f"Trajectory visualization saved to {output_path}")
