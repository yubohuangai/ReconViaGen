"""Camera I/O for EasyMocap-format calibration files.

Vendors the read path of ``FileStorage`` and ``read_camera`` from
``Motion-Capture/easymocap/mytools/camera_utils.py`` so that the
classical-MVS pipeline has no cross-repo import dependency.

Calibration convention (OpenCV world-to-camera):
    X_c = R @ X_w + T
"""

import os
from os.path import join
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Minimal OpenCV-YAML reader (vendored from EasyMocap FileStorage read path)
# ---------------------------------------------------------------------------

class _FileStorage:
    """Read-only wrapper around ``cv2.FileStorage``."""

    def __init__(self, filename: str):
        assert os.path.exists(filename), f"File not found: {filename}"
        self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    def __del__(self):
        cv2.FileStorage.release(self.fs)

    def read(self, key: str, dt: str = "mat"):
        if dt == "mat":
            return self.fs.getNode(key).mat()
        elif dt == "list":
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == "":
                    val = str(int(n.at(i).real()))
                if val != "none":
                    results.append(val)
            return results
        elif dt == "int":
            node = self.fs.getNode(key)
            if node.empty():
                return None
            return int(node.real())
        raise NotImplementedError(f"dt={dt!r}")


# ---------------------------------------------------------------------------
# Camera loading
# ---------------------------------------------------------------------------

def read_cameras(
    intri_path: str,
    extri_path: str,
) -> Tuple[Dict[str, dict], List[str]]:
    """Load all cameras from EasyMocap-format YAML files.

    Returns
    -------
    cams : dict
        ``cams[cam_name]`` has keys ``K``, ``R``, ``T``, ``Rvec``,
        ``dist``, ``H``, ``W``, ``RT`` (3x4), ``P`` (3x4).
    cam_names : list[str]
        Ordered camera identifiers from the ``names`` field.
    """
    assert os.path.exists(intri_path), intri_path
    assert os.path.exists(extri_path), extri_path

    intri = _FileStorage(intri_path)
    extri = _FileStorage(extri_path)

    cam_names: List[str] = intri.read("names", dt="list")
    cams: Dict[str, dict] = {}

    for cam in cam_names:
        c: dict = {}
        c["K"] = intri.read(f"K_{cam}")
        c["invK"] = np.linalg.inv(c["K"])

        H = intri.read(f"H_{cam}", dt="int")
        W = intri.read(f"W_{cam}", dt="int")
        c["H"] = H if H is not None else -1
        c["W"] = W if W is not None else -1

        Rvec = extri.read(f"R_{cam}")
        Tvec = extri.read(f"T_{cam}")
        assert Rvec is not None, f"Missing R_{cam}"
        R = cv2.Rodrigues(Rvec)[0]

        c["R"] = R
        c["Rvec"] = Rvec
        c["T"] = Tvec
        c["RT"] = np.hstack((R, Tvec))
        c["P"] = c["K"] @ c["RT"]
        # Camera center in world coords:  C = -R^T @ T
        c["center"] = -R.T @ Tvec

        dist = intri.read(f"dist_{cam}")
        if dist is None:
            dist = intri.read(f"D_{cam}")
        c["dist"] = dist
        cams[cam] = c

    return cams, cam_names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def undistort_image(
    img: np.ndarray,
    K: np.ndarray,
    dist: Optional[np.ndarray],
) -> np.ndarray:
    """Undistort *img* using intrinsic *K* and distortion coefficients."""
    if dist is None or np.allclose(dist, 0):
        return img
    return cv2.undistort(img, K, dist, None)


def cam_to_extrinsic_4x4(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Build a 4x4 world-to-camera extrinsic matrix ``[R|T ; 0 0 0 1]``."""
    ext = np.eye(4, dtype=np.float64)
    ext[:3, :3] = R
    ext[:3, 3] = T.ravel()
    return ext


def cam_to_o3d_intrinsic(K: np.ndarray, H: int, W: int):
    """Convert a 3x3 intrinsic matrix to ``open3d.camera.PinholeCameraIntrinsic``."""
    import open3d as o3d

    return o3d.camera.PinholeCameraIntrinsic(
        width=int(W),
        height=int(H),
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
    )


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def load_images(
    data_root: str,
    cam_names: List[str],
    frame: int = 0,
    ext: str = ".jpg",
) -> Dict[str, np.ndarray]:
    """Load one image per camera following the Motion-Capture layout.

    Tries ``images/<cam>/{frame:06d}{ext}`` first, then falls back to the
    first file matching ``images/<cam>/*{ext}``.
    """
    images: Dict[str, np.ndarray] = {}
    for cam in cam_names:
        cam_dir = join(data_root, "images", cam)
        primary = join(cam_dir, f"{frame:06d}{ext}")
        if os.path.exists(primary):
            path = primary
        else:
            candidates = sorted(glob(join(cam_dir, f"*{ext}")))
            if not candidates:
                raise FileNotFoundError(
                    f"No images found for camera {cam} in {cam_dir}"
                )
            path = candidates[min(frame, len(candidates) - 1)]
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"Failed to read {path}")
        images[cam] = img
    return images


def load_masks(
    data_root: str,
    cam_names: List[str],
    mask_dir: str = "masks",
    frame: int = 0,
    ext: str = ".png",
) -> Optional[Dict[str, np.ndarray]]:
    """Load optional foreground masks.  Returns *None* if the mask dir doesn't exist."""
    mask_root = join(data_root, mask_dir)
    if not os.path.isdir(mask_root):
        return None
    masks: Dict[str, np.ndarray] = {}
    for cam in cam_names:
        cam_dir = join(mask_root, cam)
        primary = join(cam_dir, f"{frame:06d}{ext}")
        if os.path.exists(primary):
            path = primary
        else:
            candidates = sorted(glob(join(cam_dir, f"*{ext}")))
            if not candidates:
                masks[cam] = None
                continue
            path = candidates[min(frame, len(candidates) - 1)]
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        masks[cam] = m
    return masks
