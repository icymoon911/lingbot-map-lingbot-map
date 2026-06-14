"""ScanNet v2 dataset loader.

Dataset format:
  Source format:
    {raw_data_root}/
      scene0000_00/
        color/
          0.jpg                       # RGB image (1296x968, JPEG)
          1.jpg
          ...
        depth/
          0.png                       # Depth map (640x480, 16-bit PNG, millimeters)
          1.png
          ...
        pose/
          0.txt                       # 4x4 C2W matrix (text)
          1.txt
          ...
        intrinsic/
          intrinsic_color.txt          # 4x4 color camera intrinsic matrix
          intrinsic_depth.txt          # 4x4 depth camera intrinsic matrix
      scannetv2_train.txt              # Train split (one scene ID per line)
      scannetv2_val.txt                # Val split (one scene ID per line)
      scannetv2_test.txt               # Test split (one scene ID per line)

Notes:
  - Color images are 1296x968; depth maps are 640x480.
    Color is resized to 640x480 to match depth; depth intrinsics are used.
  - Depth is stored as 16-bit unsigned integer PNG in millimeters.
    Converted to float32 meters. All-zero depth frames (sensor dropout) are skipped.
  - GT poses are stored as 4x4 matrices in text files, one per frame.
  - Invalid poses (containing NaN or inf) are skipped.
  - Split files follow the official ScanNet v2 convention.

Reference:
  https://github.com/ScanNet/ScanNet
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional
from benchmark.core.loader import BSSLoader
from benchmark.dataset.base import BaseDataset

# ScanNet v2 default depth intrinsics (640x480).
# Used as fallback when per-scene intrinsic files are missing.
INTRINSICS_SCANNET_DEPTH = {
    'fx': 577.87060546875,
    'fy': 579.96875,
    'cx': 319.5,
    'cy': 239.5,
    'width': 640,
    'height': 480,
}

# Evaluation thresholds for ScanNet point cloud metrics (meters).
EVAL_THRESHOLDS = [0.05, 0.10]


class ScannetDataset(BaseDataset):
    """ScanNet v2 dataset loader."""

    def __init__(
        self,
        raw_data_root: str,
        split: str = 'val',
        split_file: Optional[str] = None,
        logger=None,
    ):
        """Initialize ScanNet v2 dataset loader.

        Args:
            raw_data_root: Dataset root directory containing scene folders
                and optional split files.
            split: One of 'train', 'val', 'test' (default: 'val').
                Used to locate the split file ``scannetv2_{split}.txt`` in
                ``raw_data_root`` unless ``split_file`` is given explicitly.
            split_file: Optional explicit path to a split file. When provided
                this overrides the default ``scannetv2_{split}.txt`` lookup.
            logger: Optional logger instance.
        """
        super().__init__(raw_data_root, logger=logger)
        self.split = split
        self._split_file = split_file
        self._scene_cache: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_scenes(self) -> List[str]:
        """Get scene names, optionally filtered by split file.

        Returns:
            Sorted list of scene directory names (e.g. ``['scene0000_00', ...]``).
        """
        if self._scene_cache is not None:
            return self._scene_cache

        split_file_path = self._resolve_split_file()
        if split_file_path is not None and split_file_path.exists():
            with open(split_file_path, 'r') as f:
                allowed = {
                    line.strip() for line in f
                    if line.strip() and not line.strip().startswith('#')
                }
            self.logger and self.logger.info(
                f"ScanNet: loaded split '{self.split}' from {split_file_path} "
                f"({len(allowed)} scenes)"
            )
        else:
            allowed = None
            if split_file_path is not None:
                self.logger and self.logger.warning(
                    f"ScanNet: split file {split_file_path} not found, "
                    f"using all scene directories"
                )

        scenes = []
        for scene_dir in sorted(self.raw_data_root.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name.startswith('.'):
                continue
            # Only consider directories that look like ScanNet scenes
            if not (scene_dir / 'pose').is_dir():
                continue
            if allowed is not None and scene_dir.name not in allowed:
                continue
            scenes.append(scene_dir.name)

        self._scene_cache = scenes
        return scenes

    def get_frame_list(self, scene: str) -> List[int]:
        """Get valid frame IDs for a scene.

        A frame is valid when **all** of the following hold:
          1. Color image exists.
          2. Depth image exists and is not all-zero (sensor dropout).
          3. Pose file exists and contains a finite 4x4 matrix.

        Args:
            scene: Scene directory name (e.g. ``'scene0000_00'``).

        Returns:
            Sorted list of integer frame IDs.
        """
        scene_dir = self.raw_data_root / scene
        depth_dir = scene_dir / 'depth'
        pose_dir = scene_dir / 'pose'
        color_dir = scene_dir / 'color'

        valid_frames: List[int] = []

        for depth_file in sorted(depth_dir.glob('*.png')):
            frame_id = int(depth_file.stem)

            # Check that color and pose files exist
            color_file = self._find_color_file(color_dir, frame_id)
            pose_file = pose_dir / f"{frame_id}.txt"
            if color_file is None or not pose_file.exists():
                continue

            # Check depth is not all-zero (sensor dropout)
            depth_raw = np.array(Image.open(depth_file), dtype=np.uint16)
            if depth_raw.max() == 0:
                continue

            # Check pose is valid (finite values)
            try:
                pose = np.loadtxt(pose_file).reshape(4, 4)
                if not np.all(np.isfinite(pose)):
                    continue
            except Exception:
                continue

            valid_frames.append(frame_id)

        return sorted(valid_frames)

    def load_frame_data(self, scene: str, frame_id: int) -> Dict[str, Any]:
        """Load single frame data.

        Args:
            scene: Scene directory name.
            frame_id: Frame number (integer).

        Returns:
            Dictionary containing:
            - rgb: HxWx3 RGB image (uint8, 0-255), resized to depth resolution
            - depth: HxW depth map (float32, meters)
            - pose: 4x4 C2W transformation matrix
            - intrinsics: [fx, fy, cx, cy] array
        """
        scene_dir = self.raw_data_root / scene
        depth_dir = scene_dir / 'depth'
        color_dir = scene_dir / 'color'
        pose_dir = scene_dir / 'pose'

        # File paths
        depth_file = depth_dir / f"{frame_id}.png"
        color_file = self._find_color_file(color_dir, frame_id)
        pose_file = pose_dir / f"{frame_id}.txt"

        if color_file is None or not depth_file.exists() or not pose_file.exists():
            raise FileNotFoundError(
                f"Missing files for {scene}/frame-{frame_id}"
            )

        # Load depth first to know target resolution
        depth = self._load_depth(depth_file)
        target_h, target_w = depth.shape[:2]

        # Load RGB and resize to depth resolution
        rgb = self._load_rgb(color_file, target_w, target_h)

        # Load pose
        c2w = self._load_pose(pose_file)

        # Load intrinsics
        intrinsics = self._load_intrinsics(scene_dir, target_w, target_h)

        return {
            'rgb': rgb,
            'depth': depth,
            'pose': c2w,
            'intrinsics': intrinsics,
        }

    def load_global_data(self, scene: str) -> Dict[str, Any]:
        """Load global scene-level data.

        ScanNet does not provide pre-computed global point clouds.
        Point clouds are built from depth maps during evaluation.

        Returns:
            Empty dictionary.
        """
        return {}

    # ------------------------------------------------------------------
    # Point cloud evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_pointcloud(
        gt_loader: BSSLoader,
        pred_loader: BSSLoader,
        logger,
        options: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate point cloud reconstruction for ScanNet v2.

        Pipeline (same structure as 7Scenes / ETH3D):
          1. GT resized to pred resolution → pixel-level Umeyama alignment.
          2. ICP fine registration against full-resolution GT point cloud.
          3. Precision / recall / F1 at **two** thresholds: 0.05 m and 0.10 m.

        Args:
            gt_loader:   BSSLoader for ground truth (native resolution).
            pred_loader: BSSLoader for method output.
            logger:      Logger instance.
            options:     Optional dict; supported keys:
                           icp_threshold (float, default 0.1)
                           voxel_size (float, default 4/512)
                           conf_threshold (float, default 0.0)
                           thresholds (list[float], default [0.05, 0.10])

        Returns:
            Dictionary with point cloud metrics and point clouds, or None on failure.
        """
        from benchmark.geometry.registration import (
            umeyama_registration,
            icp_registration,
            apply_transform,
            voxel_downsample,
        )
        from benchmark.evaluation.points import evaluate_pointcloud as eval_pc

        icp_threshold = (options or {}).get('icp_threshold', 0.1)
        voxel_size = (options or {}).get('voxel_size', 4.0 / 512.0)
        conf_threshold = (options or {}).get('conf_threshold', 0.0)
        thresholds = (options or {}).get('thresholds', EVAL_THRESHOLDS)

        # --- Load point clouds for Umeyama (pixel-wise correspondences) ---
        gt_xyzrgb_for_umeyama, gt_mask_for_umeyama = gt_loader.load_point_cloud_grid()
        pred_xyzrgb, pred_mask = pred_loader.load_point_cloud_grid(
            confidence_threshold=conf_threshold,
        )

        # Align GT frame axis to pred's keyframe indices
        pred_frame_indices = pred_loader.get_frame_indices()
        gt_xyzrgb_for_umeyama = gt_xyzrgb_for_umeyama[pred_frame_indices]
        gt_mask_for_umeyama = gt_mask_for_umeyama[pred_frame_indices]

        # Pixel-wise correspondence
        common_mask = gt_mask_for_umeyama & pred_mask
        gt_pts_for_umeyama = gt_xyzrgb_for_umeyama[common_mask][:, :3]
        pred_pts_for_umeyama = pred_xyzrgb[common_mask][:, :3]

        logger.info(
            f"ScanNet eval: Umeyama with {len(gt_pts_for_umeyama):,} correspondences"
        )
        T_umeyama = umeyama_registration(
            source_points=pred_pts_for_umeyama,
            target_points=gt_pts_for_umeyama,
        )
        logger.info(f"Umeyama transform:\n{T_umeyama}")

        # --- Full GT point cloud for ICP and evaluation ---
        gt_xyzrgb, gt_mask = gt_loader.load_point_cloud_grid()
        gt_pts = gt_xyzrgb[gt_mask][:, :3]

        # Apply Umeyama to pred, then ICP
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
            f"Evaluating: {len(pred_pts_eval):,} pred points "
            f"vs {len(gt_pts_eval):,} gt points "
            f"at thresholds {thresholds}"
        )
        results = eval_pc(
            source_points=pred_pts_eval,
            target_points=gt_pts_eval,
            thresholds=thresholds,
        )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_split_file(self) -> Optional[Path]:
        """Resolve the path to the split file."""
        if self._split_file is not None:
            return Path(self._split_file)
        candidate = self.raw_data_root / f"scannetv2_{self.split}.txt"
        return candidate

    @staticmethod
    def _find_color_file(color_dir: Path, frame_id: int) -> Optional[Path]:
        """Find color image file for a given frame ID.

        ScanNet color images may be stored as .jpg, .jpeg, or .png.

        Args:
            color_dir: Path to the color image directory.
            frame_id: Frame number.

        Returns:
            Path to the color file, or None if not found.
        """
        for ext in ('.jpg', '.jpeg', '.png'):
            path = color_dir / f"{frame_id}{ext}"
            if path.exists():
                return path
        return None

    @staticmethod
    def _load_rgb(
        color_file: Path,
        target_w: int,
        target_h: int,
    ) -> np.ndarray:
        """Load RGB image and resize to target resolution.

        Args:
            color_file: Path to RGB image.
            target_w: Target width (depth resolution).
            target_h: Target height (depth resolution).

        Returns:
            HxWx3 RGB image (uint8, 0-255).
        """
        img = Image.open(color_file).convert('RGB')
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    @staticmethod
    def _load_depth(depth_file: Path) -> np.ndarray:
        """Load depth map from 16-bit PNG (millimeters → float32 meters).

        Args:
            depth_file: Path to depth PNG.

        Returns:
            HxW depth map (float32, meters).
        """
        depthmap = np.array(Image.open(depth_file), dtype=np.float32)

        # Validate shape
        expected_shape = (INTRINSICS_SCANNET_DEPTH['height'],
                          INTRINSICS_SCANNET_DEPTH['width'])
        if depthmap.shape != expected_shape:
            raise ValueError(
                f"Depth map shape {depthmap.shape} does not match "
                f"expected {expected_shape}"
            )

        # Convert millimeters → meters
        # 0 remains 0 (invalid / no measurement)
        depthmap = np.nan_to_num(depthmap, 0.0) / 1000.0

        # Filter invalid depth values
        depthmap[depthmap > 10.0] = 0    # Too far (>10m)
        depthmap[depthmap < 1e-3] = 0    # Too near (<1mm)

        return depthmap

    @staticmethod
    def _load_pose(pose_file: Path) -> np.ndarray:
        """Load pose file (4x4 C2W matrix).

        Args:
            pose_file: Path to pose text file.

        Returns:
            4x4 C2W transformation matrix (float32).
        """
        pose = np.loadtxt(pose_file).reshape(4, 4)
        return pose.astype(np.float32)

    @staticmethod
    def _load_intrinsics(
        scene_dir: Path,
        target_w: int,
        target_h: int,
    ) -> np.ndarray:
        """Load depth camera intrinsics for the scene.

        Tries to read ``intrinsic/intrinsic_depth.txt`` first. Falls back to
        the default ScanNet v2 depth intrinsics if the file is missing.

        Args:
            scene_dir: Scene root directory.
            target_w: Output width (for validation; intrinsics are always at
                the native 640x480 depth resolution).
            target_h: Output height.

        Returns:
            [fx, fy, cx, cy] array (float32).
        """
        intrinsic_file = scene_dir / 'intrinsic' / 'intrinsic_depth.txt'
        if intrinsic_file.exists():
            K = np.loadtxt(intrinsic_file).reshape(4, 4)
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
        else:
            fx = INTRINSICS_SCANNET_DEPTH['fx']
            fy = INTRINSICS_SCANNET_DEPTH['fy']
            cx = INTRINSICS_SCANNET_DEPTH['cx']
            cy = INTRINSICS_SCANNET_DEPTH['cy']

        return np.array([fx, fy, cx, cy], dtype=np.float32)
