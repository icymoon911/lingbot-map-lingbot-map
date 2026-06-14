"""ScanNet v2 dataset loader.

Dataset format:
  Source format:
    {raw_data_root}/
      scene{XXXX}_{XX}/
        color/
          0.jpg, 1.jpg, ...           # RGB images
        depth/
          0.png, 1.png, ...           # 16-bit PNG depth maps (millimeters)
        pose/
          0.txt, 1.txt, ...           # 4x4 C2W matrices
        intrinsic/
          intrinsic_color.txt          # 3x3 intrinsics matrix (color camera)
          intrinsic_depth.txt          # 3x3 intrinsics matrix (depth camera)
      scannetv2_train.txt              # Train split scene list
      scannetv2_val.txt                # Validation split scene list
      scannetv2_test.txt               # Test split scene list

Notes:
  - Depth maps are 16-bit unsigned PNG in millimeters; converted to float32 meters.
  - Frames with all-zero depth (sensor drop) are excluded from the frame list
    and will not be written to BSS.
  - Intrinsics are read per-scene from intrinsic/intrinsic_color.txt.
  - Split files (scannetv2_{train,val,test}.txt) are expected at raw_data_root.

Reference:
  https://github.com/ScanNet/ScanNet
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional
from benchmark.core.loader import BSSLoader
from benchmark.dataset.base import BaseDataset


class ScannetDataset(BaseDataset):
    """ScanNet v2 dataset loader."""

    def __init__(self, raw_data_root: str, split: str = 'val',
                 logger=None):
        """Initialize ScanNet v2 dataset loader.

        Args:
            raw_data_root: Dataset root directory (contains scene dirs + split files)
            split: 'train', 'val', or 'test' (default: 'val').
                   Uses scannetv2_{split}.txt to filter scenes.
                   If the split file is not found, all scene directories are used.
            logger: Optional logger instance
        """
        super().__init__(raw_data_root, logger=logger)
        self.split = split

    def get_scenes(self) -> List[str]:
        """Get all scene names filtered by split.

        Returns:
            List of scene directory names (e.g. ['scene0000_00', 'scene0000_01', ...])
        """
        all_scenes = sorted([
            d.name for d in self.raw_data_root.iterdir()
            if d.is_dir() and d.name.startswith('scene')
        ])

        split_file = self.raw_data_root / f"scannetv2_{self.split}.txt"
        if split_file.exists():
            with open(split_file, 'r') as f:
                split_scenes = set(line.strip() for line in f if line.strip())
            filtered = [s for s in all_scenes if s in split_scenes]
            if self.logger:
                self.logger.info(
                    f"ScanNet split '{self.split}': {len(filtered)}/{len(all_scenes)} scenes "
                    f"from {split_file.name}"
                )
            return filtered

        if self.logger:
            self.logger.warning(
                f"Split file {split_file} not found, using all {len(all_scenes)} scenes"
            )
        return all_scenes

    def get_frame_list(self, scene: str) -> List[int]:
        """Get all valid frame IDs for a scene (excludes all-zero depth frames).

        Args:
            scene: Scene name (e.g. 'scene0000_00')

        Returns:
            Sorted list of frame IDs (integers)
        """
        scene_dir = self.raw_data_root / scene
        all_frames = self._get_all_frame_ids(scene_dir)

        # Filter out frames with all-zero depth (sensor drop)
        valid_frames = [
            fid for fid in all_frames
            if self._has_valid_depth(scene_dir, fid)
        ]

        if self.logger and len(valid_frames) < len(all_frames):
            self.logger.info(
                f"Scene {scene}: filtered {len(all_frames) - len(valid_frames)} "
                f"all-zero depth frames ({len(valid_frames)}/{len(all_frames)} valid)"
            )

        return sorted(valid_frames)

    def _get_all_frame_ids(self, scene_dir: Path) -> List[int]:
        """Get all frame IDs from the color directory."""
        color_dir = scene_dir / 'color'
        frame_ids = []
        for f in sorted(color_dir.glob('*.jpg')):
            try:
                frame_ids.append(int(f.stem))
            except ValueError:
                continue
        return sorted(frame_ids)

    def _has_valid_depth(self, scene_dir: Path, frame_id: int) -> bool:
        """Check if a frame has valid (non-all-zero) depth."""
        depth_file = scene_dir / 'depth' / f"{frame_id}.png"
        if not depth_file.exists():
            return False
        depth = np.array(Image.open(depth_file), dtype=np.uint16)
        return depth.sum() > 0

    def load_frame_data(self, scene: str, frame_id: int) -> Dict[str, Any]:
        """Load single frame data.

        Args:
            scene: Scene name (e.g. 'scene0000_00')
            frame_id: Frame number (integer)

        Returns:
            Dictionary containing:
            - rgb: HxWx3 RGB image (uint8, 0-255)
            - depth: HxW depth map (float32, meters)
            - pose: 4x4 C2W transformation matrix
            - intrinsics: [fx, fy, cx, cy] array
        """
        scene_dir = self.raw_data_root / scene

        color_file = scene_dir / 'color' / f"{frame_id}.jpg"
        depth_file = scene_dir / 'depth' / f"{frame_id}.png"
        pose_file = scene_dir / 'pose' / f"{frame_id}.txt"

        if not all([color_file.exists(), depth_file.exists(), pose_file.exists()]):
            raise FileNotFoundError(
                f"Missing files for {scene}/frame-{frame_id}"
            )

        rgb = self._load_rgb(color_file)
        depth = self._load_depth(depth_file)
        c2w = self._load_pose(pose_file)
        intrinsics = self._load_intrinsics(scene_dir, rgb.shape[:2])

        return {
            'rgb': rgb,
            'depth': depth,
            'pose': c2w,
            'intrinsics': intrinsics,
        }

    def load_global_data(self, scene: str) -> Dict[str, Any]:
        """Load global scene-level data.

        Args:
            scene: Scene name

        Returns:
            Empty dictionary (ScanNet doesn't provide pre-computed global point clouds)
        """
        return {}

    @staticmethod
    def _load_rgb(color_file: Path) -> np.ndarray:
        """Load RGB image.

        Args:
            color_file: Path to RGB image file

        Returns:
            HxWx3 RGB image (uint8, 0-255)
        """
        img = Image.open(color_file).convert('RGB')
        return np.array(img, dtype=np.uint8)

    @staticmethod
    def _load_depth(depth_file: Path) -> np.ndarray:
        """Load depth map.

        ScanNet v2 depth maps are 16-bit PNG in millimeters.
        Converted to float32 meters. All-zero frames (sensor drop)
        should already be filtered by get_frame_list().

        Args:
            depth_file: Path to 16-bit PNG depth image

        Returns:
            HxW depth map (float32, meters)
        """
        depthmap = np.array(Image.open(depth_file), dtype=np.float32)

        # Convert from millimeters to meters
        depthmap = depthmap / 1000.0

        # Filter invalid depth values
        depthmap[depthmap < 1e-3] = 0.0    # Too near (<1mm)
        depthmap[~np.isfinite(depthmap)] = 0.0

        return depthmap

    @staticmethod
    def _load_pose(pose_file: Path) -> np.ndarray:
        """Load pose file (4x4 C2W matrix).

        Args:
            pose_file: Path to pose text file

        Returns:
            4x4 C2W transformation matrix (float32)
        """
        pose = np.loadtxt(pose_file).reshape(4, 4)
        return pose.astype(np.float32)

    @staticmethod
    def _load_intrinsics(scene_dir: Path, image_hw: tuple) -> np.ndarray:
        """Load camera intrinsics from per-scene intrinsic file.

        Args:
            scene_dir: Path to scene directory
            image_hw: (height, width) of loaded RGB image

        Returns:
            [fx, fy, cx, cy] array (float32)
        """
        intrinsic_file = scene_dir / 'intrinsic' / 'intrinsic_color.txt'
        if intrinsic_file.exists():
            K = np.loadtxt(intrinsic_file).reshape(4, 4)[:3, :3]
        else:
            # ScanNet v2 default color camera intrinsics
            K = np.array([
                [577.870605, 0.0, 319.5],
                [0.0, 577.870605, 239.5],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

        return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)

    @staticmethod
    def evaluate_pointcloud(
        gt_loader: BSSLoader,
        pred_loader: BSSLoader,
        logger,
        options: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Evaluate point cloud reconstruction for ScanNet v2.

        Alignment uses pixel-wise correspondences (Umeyama + ICP), same
        pipeline as SevenScenesDataset.  Evaluation uses two distance
        thresholds (0.05 m and 0.10 m) suitable for indoor scenes.

        Args:
            gt_loader:   BSSLoader for ground truth
            pred_loader: BSSLoader for method output
            logger:      Logger instance
            options:     Optional dict; supported keys:
                           icp_threshold (float, default 0.1)
                           voxel_size (float, default 4/512)
                           conf_threshold (float, default 0.0)

        Returns:
            Dictionary with point cloud metrics and point clouds, or None on failure
        """
        from benchmark.geometry.registration import (
            umeyama_registration, icp_registration, apply_transform, voxel_downsample
        )
        from benchmark.evaluation.points import evaluate_pointcloud as eval_pc

        icp_threshold = (options or {}).get('icp_threshold', 0.1)
        voxel_size = (options or {}).get('voxel_size', 4.0 / 512.0)
        conf_threshold = (options or {}).get('conf_threshold', 0.0)

        # --- Load point clouds for pixel-wise Umeyama alignment ---
        gt_xyzrgb_for_umeyama, gt_mask_for_umeyama = gt_loader.load_point_cloud_grid()
        pred_xyzrgb, pred_mask = pred_loader.load_point_cloud_grid(
            confidence_threshold=conf_threshold
        )

        # Align GT frames to pred keyframe indices (sparse SLAM support)
        pred_frame_indices = pred_loader.get_frame_indices()
        gt_xyzrgb_for_umeyama = gt_xyzrgb_for_umeyama[pred_frame_indices]
        gt_mask_for_umeyama = gt_mask_for_umeyama[pred_frame_indices]

        # Pixel-wise correspondence: only pixels valid in both grids
        common_mask = gt_mask_for_umeyama & pred_mask
        gt_pts_for_umeyama = gt_xyzrgb_for_umeyama[common_mask][:, :3]
        pred_pts_for_umeyama = pred_xyzrgb[common_mask][:, :3]

        logger.info(
            f"ScanNet: Umeyama alignment with {len(gt_pts_for_umeyama)} "
            f"corresponding points"
        )
        T_umeyama = umeyama_registration(
            source_points=pred_pts_for_umeyama,
            target_points=gt_pts_for_umeyama,
        )
        logger.info(f"Umeyama transform:\n{T_umeyama}")

        # --- Load full GT point cloud for ICP and evaluation ---
        gt_xyzrgb, gt_mask = gt_loader.load_point_cloud_grid()
        gt_pts = gt_xyzrgb[gt_mask][:, :3]

        # Apply Umeyama to all pred points
        all_pred_pts = pred_xyzrgb[common_mask][:, :3]
        all_pred_after_umeyama = apply_transform(all_pred_pts, T_umeyama)

        # Voxel downsample
        if voxel_size > 0:
            logger.info(
                f"Voxel downsampling at {voxel_size:.6f}m "
                f"(pred: {len(all_pred_after_umeyama):,}, gt: {len(gt_pts):,})"
            )
            pred_ds = voxel_downsample(all_pred_after_umeyama, voxel_size)
            gt_ds = voxel_downsample(gt_pts, voxel_size)
            logger.info(f"After downsampling: pred={len(pred_ds):,}, gt={len(gt_ds):,}")
        else:
            pred_ds = all_pred_after_umeyama
            gt_ds = gt_pts

        # ICP fine registration
        logger.info(f"ICP alignment with threshold {icp_threshold}")
        T_icp = icp_registration(
            source_points=pred_ds,
            target_points=gt_ds,
            icp_threshold=icp_threshold,
        )
        logger.info(f"ICP transform:\n{T_icp}")

        pred_pts_eval = apply_transform(pred_ds, T_icp)
        gt_pts_eval = gt_ds

        logger.info(
            f"Evaluating: {len(pred_pts_eval)} pred points "
            f"vs {len(gt_pts_eval)} gt points"
        )

        # ScanNet uses two distance thresholds: 0.05m and 0.10m
        results = eval_pc(
            source_points=pred_pts_eval,
            target_points=gt_pts_eval,
            thresholds=[0.05, 0.10],
        )

        return results
