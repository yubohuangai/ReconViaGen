"""Open3D utilities consolidated from 3D_Object_Reconstruction.

Functions in this module are adapted (with citations) from:
  - registration.py  : rigid_transform_3D, match_ransac, icp
  - utils.py         : preprocess_point_cloud
  - preprocess_pcd.py: point-cloud cleaning recipe (plane removal, DBSCAN,
                       statistical + radius outlier removal)

The originals have fragile module-level imports (``from SIFT import *``,
``from plot import ...``) that drag in RealSense / OpenCV-contrib / Kornia
dependencies.  We extract only the self-contained logic here.
"""

from typing import Optional, Tuple

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Rigid-body estimation  (from 3D_Object_Reconstruction/registration.py:213-254)
# ---------------------------------------------------------------------------

def rigid_transform_3D(
    A: np.ndarray,
    B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """SVD-based rigid transform that maps *A* to *B*.

    Parameters
    ----------
    A, B : (N, 3) arrays of corresponding 3-D points.

    Returns
    -------
    R : (3, 3) rotation matrix.
    t : (3,) translation vector such that ``B ≈ R @ A.T + t``.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    assert A.shape == B.shape and A.shape[1] == 3

    cA = A.mean(axis=0)
    cB = B.mean(axis=0)
    H = (A - cA).T @ (B - cB)
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = cB - R @ cA
    return R, t


# ---------------------------------------------------------------------------
# RANSAC rigid fit  (from 3D_Object_Reconstruction/registration.py:163-211)
# ---------------------------------------------------------------------------

def match_ransac(
    p: np.ndarray,
    p_prime: np.ndarray,
    tol: float = 0.01,
) -> Optional[np.ndarray]:
    """Estimate a 4x4 rigid transform from *p* to *p_prime* via RANSAC.

    Returns the homogeneous transform or ``None`` if RMSE > *tol*.
    """
    assert len(p) == len(p_prime)
    R, t = rigid_transform_3D(p, p_prime)
    transformed = (R @ p.T).T + t
    errors = np.linalg.norm(transformed - p_prime, axis=1)
    k = max(1, int(len(p) * 0.7))
    rmse = errors[np.argpartition(errors, k)[:k]].mean()
    if rmse < tol:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    return None


# ---------------------------------------------------------------------------
# ICP  (from 3D_Object_Reconstruction/registration.py:13-69)
# ---------------------------------------------------------------------------

def icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float = 0.005,
    max_correspondence_distance_coarse: float = 0.05,
    max_correspondence_distance_fine: float = 0.01,
    method: str = "point-to-plane",
) -> Tuple[np.ndarray, "o3d.pipelines.registration.RegistrationResult"]:
    """Run coarse-to-fine ICP or colored ICP.

    Returns ``(4x4 transform, information_matrix)``.
    """
    assert method in ("point-to-plane", "colored-icp")

    if method == "point-to-plane":
        res_coarse = o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance_coarse,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        res = o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance_fine,
            res_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        T = res.transformation
    else:
        res = o3d.pipelines.registration.registration_colored_icp(
            source, target,
            voxel_size,
            np.eye(4),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=50
            ),
        )
        T = res.transformation

    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine, T,
    )
    return T, info


# ---------------------------------------------------------------------------
# Point-cloud preprocessing  (from 3D_Object_Reconstruction/utils.py:37-50)
# ---------------------------------------------------------------------------

def preprocess_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """Voxel-downsample, estimate normals, compute FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )
    return pcd_down, fpfh


# ---------------------------------------------------------------------------
# Point-cloud cleaning  (generalised from 3D_Object_Reconstruction/preprocess_pcd.py:24-84)
# ---------------------------------------------------------------------------

def clean_point_cloud(
    pcd: o3d.geometry.PointCloud,
    *,
    remove_plane: bool = False,
    plane_dist_thresh: float = 0.02,
    dbscan_eps: float = 0.02,
    dbscan_min_points: int = 500,
    stat_nb_neighbors: int = 50,
    stat_std_ratio: float = 1.0,
    radius_nb_points: int = 30,
    radius: float = 0.01,
) -> o3d.geometry.PointCloud:
    """Progressively clean a point cloud.

    Steps (each optional / configurable):
      1. RANSAC plane removal (e.g. table/floor)
      2. DBSCAN clustering – keep the largest cluster
      3. Statistical outlier removal
      4. Radius outlier removal
    """
    cloud = pcd

    if remove_plane:
        _model, inliers = cloud.segment_plane(
            distance_threshold=plane_dist_thresh, ransac_n=3,
            num_iterations=1000,
        )
        cloud = cloud.select_by_index(inliers, invert=True)

    if dbscan_eps > 0 and dbscan_min_points > 0 and len(cloud.points) > dbscan_min_points:
        labels = np.asarray(
            cloud.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points)
        )
        if labels.max() >= 0:
            largest = np.argmax(np.bincount(labels[labels >= 0]))
            cloud = cloud.select_by_index(np.where(labels == largest)[0])

    if stat_nb_neighbors > 0 and len(cloud.points) > stat_nb_neighbors:
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=stat_nb_neighbors, std_ratio=stat_std_ratio,
        )

    if radius_nb_points > 0 and len(cloud.points) > radius_nb_points:
        cloud, _ = cloud.remove_radius_outlier(
            nb_points=radius_nb_points, radius=radius,
        )

    return cloud
