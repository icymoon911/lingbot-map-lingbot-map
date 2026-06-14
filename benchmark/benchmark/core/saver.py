"""BSS data saver.

Handles saving of frame and global data to BSS directory structure.
Used by both dataset prepare and method run phases.

Spatial outputs (rgb, depth, mask, confidence, points) are automatically
resized to GT resolution before saving, so all stored files are at the
same resolution as the ground truth.

Built-in spatial types are dispatched through a registry
(``_SPATIAL_TYPE_REGISTRY``), making it straightforward to add new data
types (e.g. semantic segmentation, surface normals) by registering a
save function, interpolation mode, and optional visualization helper.
"""

import cv2
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from benchmark.io.image import save_rgb, save_mask, save_exr
from benchmark.io.trajectory import write_trajectory
from benchmark.io.intrinsics import write_intrinsics
from benchmark.io.pointcloud import save_point_cloud_ply
from benchmark.io.sampling import write_sampling_json
from benchmark.utils.visualization import save_depth_visualization
from benchmark.core.storage import BSSArtifact


# ---------------------------------------------------------------------------
# Spatial type registry
# ---------------------------------------------------------------------------

# Built-in frame types that are NOT spatial per-frame images/maps
_BUILTIN_NON_SPATIAL = {'pose', 'intrinsics'}

# All built-in frame types (spatial + non-spatial)
BUILTIN_FRAME_TYPES = {'rgb', 'depth', 'mask', 'pose', 'intrinsics', 'points', 'confidence'}
BUILTIN_GLOBAL_TYPES = {'points'}


def _save_as_image(data: np.ndarray, path: Path) -> None:
    """Save RGB image via save_rgb."""
    save_rgb(data, path)


def _save_as_mask(data: np.ndarray, path: Path) -> None:
    """Save mask via save_mask."""
    save_mask(data, path)


def _save_as_exr(data: np.ndarray, path: Path) -> None:
    """Save float data via save_exr."""
    save_exr(data, path)


def _viz_depth(depth: np.ndarray, jpg_path) -> None:
    """Create percentile-based depth visualization JPG."""
    valid = (depth > 0) & np.isfinite(depth)
    if np.any(valid):
        min_d, max_d = np.percentile(depth[valid], [1, 99])
        save_depth_visualization(depth, jpg_path, min_d, max_d)


def _viz_conf(confidence: np.ndarray, jpg_path) -> None:
    """Create percentile-based confidence visualization JPG."""
    valid = (confidence > 0) & np.isfinite(confidence)
    if np.any(valid):
        min_c, max_c = np.percentile(confidence[valid], [1, 99])
        save_depth_visualization(confidence, jpg_path, min_c, max_c)


# Registry entries: (save_fn, interpolation_flag, vis_fn_or_None, file_extension)
_SPATIAL_TYPE_REGISTRY: Dict[str, Tuple[Callable, int, Optional[Callable], str]] = {
    'rgb':        (_save_as_image, cv2.INTER_LINEAR,  None,       'png'),
    'depth':      (_save_as_exr,   cv2.INTER_NEAREST, _viz_depth, 'exr'),
    'mask':       (_save_as_mask,  cv2.INTER_NEAREST, None,       'png'),
    'confidence': (_save_as_exr,   cv2.INTER_LINEAR,  _viz_conf,  'exr'),
    'points':     (_save_as_exr,   cv2.INTER_NEAREST, None,       'exr'),
}


class BSSSaver:
    """Manages saving of dataset/method frame and global data to BSS format.

    Tracks which frame and global keys were saved during the current session,
    making it possible to record them in .complete.json metadata.

    All spatial per-frame outputs are resized to GT resolution (image_width x
    image_height passed to save_frame_data) before writing to disk, ensuring
    that stored files always match the ground truth image dimensions.

    Built-in spatial types (rgb, depth, mask, confidence, points) and custom
    types are dispatched through a unified registry mechanism.
    """

    # Re-export for backward compatibility
    BUILTIN_FRAME_TYPES = BUILTIN_FRAME_TYPES
    BUILTIN_GLOBAL_TYPES = BUILTIN_GLOBAL_TYPES

    def __init__(
        self,
        artifact: BSSArtifact,
        context=None,
        logger=None,
    ):
        """Initialize BSS saver.

        Args:
            artifact:  BSSArtifact describing the output directory
            context:   Optional dataset or method instance for custom saver dispatch
            logger:    Optional logger for progress messages
        """
        self.artifact = artifact
        self.context = context
        self.logger = logger
        self.artifact.root.mkdir(parents=True, exist_ok=True)

        # Discovered custom saver methods on context
        self.custom_savers: Dict[str, Callable] = {}
        if self.context is not None:
            self._discover_custom_savers()

        # Keys saved during this session (populated during save calls)
        self._frame_keys: Set[str] = set()
        self._global_keys: Set[str] = set()
        # Original GT frame indices for sparse outputs (None = dense/identity)
        self._frame_indices: Optional[List[int]] = None

        # GT and method output dimensions (set at save_frame_data call time)
        self._gt_width: Optional[int] = None
        self._gt_height: Optional[int] = None
        self._method_width: Optional[int] = None
        self._method_height: Optional[int] = None

    def _discover_custom_savers(self) -> None:
        """Discover __save_{key}_file__ methods on context."""
        for attr_name in dir(self.context):
            if attr_name.startswith('__save_') and attr_name.endswith('_file__'):
                key = attr_name[7:-7]
                method = getattr(self.context, attr_name)
                if callable(method):
                    self.custom_savers[key] = method

    def get_completion_metadata(self) -> Dict[str, Any]:
        """Return metadata dict to be merged into .complete.json.

        Returns:
            Dict with 'frame_keys', 'global_keys' lists (sorted), and optionally
            'frame_index_map' for sparse SLAM outputs where K < N.
        """
        meta: Dict[str, Any] = {
            'frame_keys': sorted(self._frame_keys),
            'global_keys': sorted(self._global_keys),
        }
        if self._frame_indices is not None:
            meta['frame_index_map'] = list(self._frame_indices)
        return meta

    def _detect_method_size(
        self, frame_data_list: List[Dict[str, Any]]
    ) -> Optional[Tuple[int, int]]:
        """Detect method output image size from the first non-None RGB frame.

        Args:
            frame_data_list: List of frame data dictionaries

        Returns:
            (width, height) of the method output, or None if not determinable
        """
        for frame_data in frame_data_list:
            rgb = frame_data.get('rgb')
            if rgb is not None and isinstance(rgb, np.ndarray) and rgb.ndim >= 2:
                h, w = rgb.shape[:2]
                return w, h
        return None

    def _resize_to_gt(self, data: np.ndarray, interpolation: int) -> np.ndarray:
        """Resize spatial data to GT resolution if it differs from method resolution.

        Args:
            data:          Input array (HxW or HxWxC)
            interpolation: OpenCV interpolation flag

        Returns:
            Array resized to (self._gt_height, self._gt_width) if needed,
            otherwise the original array unchanged.
        """
        if (self._gt_width is None or self._method_width is None):
            return data
        if (data.shape[1] == self._gt_width and data.shape[0] == self._gt_height):
            return data
        return cv2.resize(data, (self._gt_width, self._gt_height),
                          interpolation=interpolation)

    def save_frame_data(
        self,
        frame_data_list: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        frame_indices: Optional[List[int]] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        """Save frame data for all frames in a scene (with parallel I/O).

        Frame files are named using zero-padded 6-digit indices (e.g., 000000.png),
        always contiguous from 0 to K-1 regardless of original GT frame positions.

        Spatial outputs (rgb, depth, mask, confidence, points) are automatically
        resized to (image_width x image_height) before saving.

        Args:
            frame_data_list: List of K frame data dictionaries. Each dict may contain
                             'rgb', 'depth', 'pose', 'intrinsics', 'mask',
                             'confidence', 'points', or custom keys.
            image_width:     GT image width; all spatial outputs are resized to this.
            image_height:    GT image height; all spatial outputs are resized to this.
            frame_indices:   Optional list of K original GT frame indices.
                             When provided (sparse SLAM outputs), these are written as
                             the frame_idx column in traj.txt and stored as
                             'frame_index_map' in .complete.json so evaluators can
                             select the matching GT poses.
                             When None (dense methods), identity mapping is assumed.
            max_workers:     Maximum number of parallel workers (default: CPU count)
        """
        self._frame_indices = frame_indices
        self._gt_width = image_width
        self._gt_height = image_height

        method_size = self._detect_method_size(frame_data_list)
        self._method_width = method_size[0] if method_size else None
        self._method_height = method_size[1] if method_size else None

        if (self._method_width is not None and
                (self._method_width != self._gt_width or
                 self._method_height != self._gt_height)):
            if self.logger:
                self.logger.info(
                    f"Resizing method outputs from "
                    f"{self._method_width}x{self._method_height} to "
                    f"{self._gt_width}x{self._gt_height} (GT resolution)"
                )

        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 32)

        required_dirs: Set[str] = set()
        custom_keys: Set[str] = set()

        for frame_data in frame_data_list:
            for key in _SPATIAL_TYPE_REGISTRY:
                if key in frame_data and frame_data[key] is not None:
                    required_dirs.add(key)

            for key in frame_data.keys():
                if key not in BUILTIN_FRAME_TYPES:
                    custom_keys.add(key)

        for dir_name in required_dirs:
            (self.artifact.root / dir_name).mkdir(exist_ok=True)
        for key in custom_keys:
            (self.artifact.root / key).mkdir(parents=True, exist_ok=True)

        # Collect poses and intrinsics as lists (None for missing frames)
        poses_list: List[Optional[np.ndarray]] = [None] * len(frame_data_list)
        intrinsics_list: List[Optional[np.ndarray]] = [None] * len(frame_data_list)

        save_tasks: List[Tuple[Callable, Tuple]] = []

        for idx, frame_data in enumerate(frame_data_list):
            frame_key = f"{idx:06d}"

            # Dispatch spatial types through unified registry
            for key in _SPATIAL_TYPE_REGISTRY:
                if key in frame_data and frame_data[key] is not None:
                    save_tasks.append(
                        (self._save_spatial_data, (key, frame_key, frame_data[key]))
                    )
                    self._frame_keys.add(key)

            if 'pose' in frame_data:
                poses_list[idx] = frame_data['pose']  # may be None

            if 'intrinsics' in frame_data:
                intr = frame_data['intrinsics']
                if intr is not None:
                    # Convert 3x3 matrix to [fx, fy, cx, cy] if needed
                    if hasattr(intr, 'shape') and np.asarray(intr).shape == (3, 3):
                        intr = np.array([intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]])
                    intrinsics_list[idx] = np.asarray(intr, dtype=np.float64)

            for key, value in frame_data.items():
                if key not in BUILTIN_FRAME_TYPES and value is not None:
                    save_tasks.append((self._save_custom_frame_data, (key, frame_key, value)))
                    self._frame_keys.add(key)

        if self.logger:
            self.logger.info(
                f"Starting parallel save of {len(save_tasks)} tasks with {max_workers} workers..."
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(func, *args): (func, args) for func, args in save_tasks}

            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                func, args = futures[future]
                try:
                    future.result()
                    completed += 1
                    if self.logger and (
                        completed % 500 == 0 or completed % max(1, total // 10) == 0
                    ):
                        self.logger.info(
                            f"Progress: {completed}/{total} tasks completed "
                            f"({100*completed//total}%)"
                        )
                except Exception as e:
                    func_name = func.__name__
                    if self.logger:
                        self.logger.error(f"Failed task: {func_name}{args}")
                    raise RuntimeError(f"Failed to execute {func_name}{args}: {e}") from e

        if self.logger:
            self.logger.info(f"Completed all {total} save tasks")

        # Write trajectory if any frame has a non-None pose
        has_any_pose = any(p is not None for p in poses_list)
        if has_any_pose:
            self._frame_keys.add('pose')
            write_trajectory(self.artifact.traj_file, poses_list)

        # Scale intrinsics to GT resolution before writing
        has_any_intrinsics = any(intr is not None for intr in intrinsics_list)
        if has_any_intrinsics:
            if (self._method_width is not None and
                    self._method_width != self._gt_width):
                sx = self._gt_width / self._method_width
                sy = self._gt_height / self._method_height
                for i in range(len(intrinsics_list)):
                    if intrinsics_list[i] is not None:
                        intr = intrinsics_list[i]
                        intrinsics_list[i] = np.array(
                            [intr[0] * sx, intr[1] * sy,
                             intr[2] * sx, intr[3] * sy]
                        )
            self._frame_keys.add('intrinsics')
            write_intrinsics(
                self.artifact.intrinsics_file, intrinsics_list, image_width, image_height
            )

    def save_global_data(self, global_data: Dict[str, Any]) -> None:
        """Save global scene-level data.

        Args:
            global_data: Dictionary from load_global_data()
        """
        if 'points' in global_data:
            self._save_pointcloud(global_data['points'])
            self._global_keys.add('points')

        for key, value in global_data.items():
            if key not in BUILTIN_GLOBAL_TYPES:
                self._save_custom_global_data(key, value)
                self._global_keys.add(key)

    def save_sampling_metadata(
        self,
        frame_ids: List[int],
        sampling_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save sampling metadata to sampling.json."""
        write_sampling_json(self.artifact.sampling_file, frame_ids, sampling_config)

    # ------------------------------------------------------------------
    # Unified spatial data saver (registry-based dispatch)
    # ------------------------------------------------------------------

    def _save_spatial_data(self, data_type: str, base_name: str,
                           data: np.ndarray) -> None:
        """Save spatial data through the unified type registry.

        Handles resize to GT resolution, primary file saving, and optional
        visualization JPG generation.  Both built-in and user-registered
        types go through this same dispatch path.

        Args:
            data_type: Registry key (e.g. ``'rgb'``, ``'depth'``).
            base_name: Zero-padded frame index string (e.g. ``'000042'``).
            data:      Spatial data array (HxW, HxWxC).
        """
        if data_type not in _SPATIAL_TYPE_REGISTRY:
            raise ValueError(
                f"Unknown spatial data type '{data_type}'. "
                f"Registered types: {sorted(_SPATIAL_TYPE_REGISTRY.keys())}"
            )

        save_fn, interpolation, vis_fn, ext = _SPATIAL_TYPE_REGISTRY[data_type]

        data = self._resize_to_gt(data, interpolation)

        # Save primary file
        dir_path = self.artifact.root / data_type
        save_fn(data, dir_path / f"{base_name}.{ext}")

        # Save optional visualization
        if vis_fn is not None:
            vis_fn(data, dir_path / f"{base_name}.jpg")

    # ------------------------------------------------------------------
    # Legacy per-type save helpers (delegate to registry)
    # ------------------------------------------------------------------

    def _save_rgb(self, base_name: str, rgb: np.ndarray) -> None:
        """Save RGB image, resized to GT resolution."""
        self._save_spatial_data('rgb', base_name, rgb)

    def _save_depth(self, base_name: str, depth: np.ndarray) -> None:
        """Save depth map, resized to GT resolution with nearest-neighbor."""
        if depth is None:
            if self.logger:
                self.logger.warning(f"Depth data for frame {base_name} is None, skipping save.")
            return
        self._save_spatial_data('depth', base_name, depth)

    def _save_mask(self, base_name: str, mask: np.ndarray) -> None:
        """Save mask image, resized to GT resolution with nearest-neighbor."""
        if mask is None:
            if self.logger:
                self.logger.warning(f"Mask data for frame {base_name} is None, skipping save.")
            return
        self._save_spatial_data('mask', base_name, mask)

    def _save_confidence(self, base_name: str, confidence: np.ndarray) -> None:
        """Save confidence map as EXR, resized to GT resolution."""
        if confidence is None:
            if self.logger:
                self.logger.warning(
                    f"Confidence data for frame {base_name} is None, skipping save."
                )
            return
        self._save_spatial_data('confidence', base_name, confidence)

    def _save_points(self, base_name: str, points: np.ndarray) -> None:
        """Save per-frame world-coordinate point grid as EXR, resized to GT resolution."""
        if points is None:
            if self.logger:
                self.logger.warning(f"Points data for frame {base_name} is None, skipping save.")
            return

        if not isinstance(points, np.ndarray):
            raise ValueError("Points must be a numpy array")

        if points.ndim < 2 or points.shape[-1] != 3:
            raise ValueError(f"Points must have shape [H, W, 3], got {points.shape}")

        self._save_spatial_data('points', base_name, points)

    def _save_pointcloud(self, points: np.ndarray) -> None:
        """Save global point cloud."""
        if not isinstance(points, np.ndarray):
            raise ValueError("Points must be a numpy array")

        if points.ndim < 2 or points.shape[-1] not in [3, 6]:
            raise ValueError(f"Points must have shape [..., 3] or [..., 6], got {points.shape}")

        save_point_cloud_ply(points, self.artifact.global_points_file)

    def _save_custom_frame_data(self, key: str, base_name: str, data: Any) -> None:
        """Save custom frame data using context's __save_{key}_file__ method."""
        if key not in self.custom_savers:
            raise ValueError(
                f"No saver method found for custom key '{key}'. "
                f"{self.context.__class__.__name__} must implement __save_{key}_file__ method."
            )

        custom_dir = self.artifact.root / key
        self.custom_savers[key](custom_dir, base_name, data)

    def _save_custom_global_data(self, key: str, data: Any) -> None:
        """Save custom global data using context's __save_{key}_file__ method."""
        if key not in self.custom_savers:
            raise ValueError(
                f"No saver method found for custom key '{key}'. "
                f"{self.context.__class__.__name__} must implement __save_{key}_file__ method."
            )

        self.custom_savers[key](self.artifact.root, None, data)
