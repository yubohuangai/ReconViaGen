"""TSDF volumetric fusion via Open3D.

Integrates per-view MVS depth maps (+ RGB) into a ``ScalableTSDFVolume``,
extracts a triangle mesh with vertex colours, and optionally cleans it
with the routines adapted from 3D_Object_Reconstruction.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh

from .camera_io import cam_to_extrinsic_4x4, cam_to_o3d_intrinsic
from .o3d_utils import clean_point_cloud


# ---------------------------------------------------------------------------
# Volume-bounds helper
# ---------------------------------------------------------------------------

def _estimate_volume_bounds(
    cams: Dict[str, dict],
    cam_names: List[str],
    depth_maps: Dict[str, np.ndarray],
    pad_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute an axis-aligned bounding box (in world coords) that encloses
    all back-projected depth points, padded by *pad_ratio*."""
    all_pts = []
    for name in cam_names:
        dm = depth_maps[name]
        if dm.max() <= 0:
            continue
        K = cams[name]["K"]
        R = cams[name]["R"]
        T = cams[name]["T"]
        invK = np.linalg.inv(K)

        H, W = dm.shape
        # Sparse sampling (every 8th pixel) for speed
        step = 8
        ys, xs = np.mgrid[0:H:step, 0:W:step].astype(np.float64)
        zs = dm[0:H:step, 0:W:step]
        valid = zs > 0
        if valid.sum() == 0:
            continue
        xs, ys, zs = xs[valid], ys[valid], zs[valid]
        ones = np.ones_like(xs)
        pixels = np.stack([xs, ys, ones], axis=-1)
        rays_cam = (invK @ pixels.reshape(-1, 3).T).T
        pts_cam = rays_cam * zs.reshape(-1, 1)
        pts_world = (R.T @ (pts_cam.T - T)).T
        all_pts.append(pts_world)

    if not all_pts:
        # Fallback: use camera centres
        centers = np.array(
            [(-cams[c]["R"].T @ cams[c]["T"]).ravel() for c in cam_names]
        )
        mn = centers.min(axis=0) - 0.5
        mx = centers.max(axis=0) + 0.5
        return mn, mx

    pts = np.concatenate(all_pts, axis=0)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    extent = mx - mn
    mn -= extent * pad_ratio
    mx += extent * pad_ratio
    return mn, mx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fuse_tsdf(
    images: Dict[str, np.ndarray],
    depth_maps: Dict[str, np.ndarray],
    cams: Dict[str, dict],
    cam_names: List[str],
    *,
    voxel_length: Optional[float] = None,
    sdf_trunc_factor: float = 3.0,
    depth_trunc: Optional[float] = None,
    masks: Optional[Dict[str, np.ndarray]] = None,
    clean: bool = True,
    clean_kwargs: Optional[dict] = None,
) -> trimesh.Trimesh:
    """Fuse depth maps into a mesh via Open3D TSDF.

    Parameters
    ----------
    images      : cam_name -> BGR uint8 (H,W,3)
    depth_maps  : cam_name -> (H,W) float depth (camera-frame Z, metres)
    cams        : cam_name -> {K, R, T, H, W, ...}
    cam_names   : ordered camera names
    voxel_length: TSDF voxel size in world units (auto-computed if None)
    sdf_trunc_factor : SDF truncation = factor * voxel_length
    depth_trunc : max depth to integrate (auto from depth maps if None)
    masks       : optional foreground masks (>0 = foreground)
    clean       : apply point-cloud cleaning after extraction
    clean_kwargs: extra kwargs for ``clean_point_cloud``

    Returns
    -------
    trimesh.Trimesh with vertex colours.
    """
    # Determine depth truncation
    if depth_trunc is None:
        all_d = [dm[dm > 0] for dm in depth_maps.values()]
        if all_d and len(np.concatenate(all_d)) > 0:
            depth_trunc = float(np.concatenate(all_d).max()) * 1.1
        else:
            depth_trunc = 5.0

    # Auto voxel length from volume extent
    if voxel_length is None:
        mn, mx = _estimate_volume_bounds(cams, cam_names, depth_maps)
        extent = (mx - mn).max()
        voxel_length = extent / 256.0
        print(f"[tsdf] auto voxel_length = {voxel_length:.6f}  "
              f"(extent={extent:.4f})")

    sdf_trunc = sdf_trunc_factor * voxel_length
    print(f"[tsdf] voxel_length={voxel_length:.6f}, sdf_trunc={sdf_trunc:.6f}, "
          f"depth_trunc={depth_trunc:.4f}")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for name in cam_names:
        dm = depth_maps[name]
        img_bgr = images[name]
        H, W = dm.shape[:2]
        cam = cams[name]

        # Detect H/W from image if calibration metadata missing
        real_H = cam["H"] if cam["H"] > 0 else H
        real_W = cam["W"] if cam["W"] > 0 else W

        # Zero out depth where mask says background
        if masks is not None and masks.get(name) is not None:
            mask = masks[name]
            if mask.shape[:2] != dm.shape[:2]:
                import cv2
                mask = cv2.resize(mask, (dm.shape[1], dm.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            dm = dm.copy()
            dm[mask < 128] = 0.0

        if dm.max() <= 0:
            continue

        # Open3D expects float32 depth in metres, and RGB (not BGR)
        depth_o3d = o3d.geometry.Image(dm.astype(np.float32))
        color_o3d = o3d.geometry.Image(
            np.ascontiguousarray(img_bgr[:, :, ::-1]).astype(np.uint8)
        )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        intrinsic = cam_to_o3d_intrinsic(cam["K"], real_H, real_W)
        extrinsic = cam_to_extrinsic_4x4(cam["R"], cam["T"])

        volume.integrate(rgbd, intrinsic, extrinsic)

    print("[tsdf] extracting triangle mesh …")
    mesh_o3d = volume.extract_triangle_mesh()
    mesh_o3d.compute_vertex_normals()

    # Optional cleaning pass
    if clean:
        kwargs = dict(
            stat_nb_neighbors=50,
            stat_std_ratio=1.5,
            radius_nb_points=30,
            radius=voxel_length * 5,
        )
        if clean_kwargs:
            kwargs.update(clean_kwargs)

        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh_o3d.vertices
        pcd.colors = mesh_o3d.vertex_colors
        pcd.normals = mesh_o3d.vertex_normals
        pcd_clean = clean_point_cloud(pcd, **kwargs)

        # Poisson surface reconstruction on the cleaned cloud gives a
        # watertight mesh, but requires good normals.  We fall back to
        # the TSDF mesh with cleaned vertices if Poisson fails.
        try:
            poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd_clean, depth=9,
            )
            # Remove low-density faces (artefacts at boundary)
            densities = np.asarray(densities)
            thresh = np.quantile(densities, 0.05)
            faces_to_keep = densities[np.asarray(poisson_mesh.triangles)].mean(axis=1) > thresh
            poisson_mesh.remove_triangles_by_mask(~faces_to_keep)
            poisson_mesh.remove_unreferenced_vertices()
            poisson_mesh.compute_vertex_normals()
            mesh_o3d = poisson_mesh

            # Transfer vertex colours from cleaned cloud via nearest-neighbour
            clean_tree = o3d.geometry.KDTreeFlann(pcd_clean)
            colours = np.zeros((len(mesh_o3d.vertices), 3))
            for i, v in enumerate(np.asarray(mesh_o3d.vertices)):
                _, idx, _ = clean_tree.search_knn_vector_3d(v, 1)
                colours[i] = np.asarray(pcd_clean.colors)[idx[0]]
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colours)
        except Exception as e:
            print(f"[tsdf] Poisson reconstruction failed ({e}); "
                  "using TSDF mesh directly")

    # Convert Open3D mesh -> trimesh.Trimesh
    verts = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    vert_colors = np.asarray(mesh_o3d.vertex_colors)
    # trimesh expects uint8 RGBA
    vc_uint8 = (np.clip(vert_colors, 0, 1) * 255).astype(np.uint8)
    vc_rgba = np.hstack([vc_uint8, np.full((len(vc_uint8), 1), 255, dtype=np.uint8)])

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_colors=vc_rgba,
        process=False,
    )

    print(f"[tsdf] mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh
