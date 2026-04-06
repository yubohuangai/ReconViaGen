"""Vertex-colour refinement and optional texture-bake for the fused mesh.

Vertex colours come for free from TSDF fusion, but can be improved by
re-projecting mesh vertices into the original calibrated views with
visibility-aware weighted blending.

The optional texture-bake path follows the pattern in
``trellis/utils/postprocessing_utils.py:bake_texture`` (xatlas UV + per-
texel projection), adapted to use real calibrated cameras instead of
synthetic orbit views.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh

from .camera_io import cam_to_extrinsic_4x4


# ---------------------------------------------------------------------------
# Vertex-colour refinement
# ---------------------------------------------------------------------------

def refine_vertex_colors(
    mesh: trimesh.Trimesh,
    images: Dict[str, np.ndarray],
    cams: Dict[str, dict],
    cam_names: List[str],
    masks: Optional[Dict[str, np.ndarray]] = None,
) -> trimesh.Trimesh:
    """Re-project mesh vertices into calibrated views, blend colours.

    Weights each view by ``cos(viewing_angle)`` — the dot product between
    the vertex normal and the viewing direction.  Only front-facing views
    contribute (cosine > 0).
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    N = len(verts)

    colour_sum = np.zeros((N, 3), dtype=np.float64)
    weight_sum = np.zeros((N, 1), dtype=np.float64)

    for name in cam_names:
        K = cams[name]["K"].astype(np.float64)
        R = cams[name]["R"].astype(np.float64)
        T = cams[name]["T"].astype(np.float64)
        img = images[name]
        H, W = img.shape[:2]

        # Project vertices into this camera
        pts_cam = (R @ verts.T + T).T  # (N, 3)
        z = pts_cam[:, 2]
        visible = z > 0

        proj = (K @ pts_cam.T).T
        u = proj[:, 0] / (z + 1e-8)
        v = proj[:, 1] / (z + 1e-8)

        in_bounds = visible & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)

        # Cosine weighting: normal · viewing_direction
        cam_center = (-R.T @ T).ravel()
        view_dirs = cam_center[None, :] - verts  # (N, 3)
        view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-12
        cos_angle = (normals * view_dirs).sum(axis=1)
        cos_angle = np.clip(cos_angle, 0, 1)

        mask_ok = np.ones(N, dtype=bool)
        if masks is not None and masks.get(name) is not None:
            m = masks[name]
            ui = np.clip(np.round(u).astype(int), 0, W - 1)
            vi = np.clip(np.round(v).astype(int), 0, H - 1)
            if m.shape[:2] != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_ok = m[vi, ui] > 128

        keep = in_bounds & (cos_angle > 0.05) & mask_ok
        if keep.sum() == 0:
            continue

        # Bilinear sample colours (BGR -> RGB)
        ui_f = u[keep].astype(np.float32)
        vi_f = v[keep].astype(np.float32)
        u0 = np.floor(ui_f).astype(int)
        v0 = np.floor(vi_f).astype(int)
        u1 = np.minimum(u0 + 1, W - 1)
        v1 = np.minimum(v0 + 1, H - 1)
        du = (ui_f - u0)[:, None]
        dv = (vi_f - v0)[:, None]

        c00 = img[v0, u0].astype(np.float64)[:, ::-1]
        c01 = img[v0, u1].astype(np.float64)[:, ::-1]
        c10 = img[v1, u0].astype(np.float64)[:, ::-1]
        c11 = img[v1, u1].astype(np.float64)[:, ::-1]
        colour = (c00 * (1 - du) * (1 - dv) +
                  c01 * du * (1 - dv) +
                  c10 * (1 - du) * dv +
                  c11 * du * dv)

        w = cos_angle[keep, None]
        colour_sum[keep] += colour * w
        weight_sum[keep] += w

    valid = weight_sum.ravel() > 0
    colours = np.zeros((N, 3), dtype=np.float64)
    colours[valid] = colour_sum[valid] / weight_sum[valid]
    colours[~valid] = np.asarray(mesh.visual.vertex_colors[:, :3],
                                 dtype=np.float64)[~valid]

    vc_uint8 = np.clip(colours, 0, 255).astype(np.uint8)
    vc_rgba = np.hstack([vc_uint8, np.full((N, 1), 255, dtype=np.uint8)])

    mesh_out = mesh.copy()
    mesh_out.visual.vertex_colors = vc_rgba
    return mesh_out


# ---------------------------------------------------------------------------
# Texture-bake (optional, uses xatlas)
# ---------------------------------------------------------------------------

def bake_texture(
    mesh: trimesh.Trimesh,
    images: Dict[str, np.ndarray],
    cams: Dict[str, dict],
    cam_names: List[str],
    texture_size: int = 2048,
    masks: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """UV-parameterise mesh with xatlas, then project calibrated views
    into a texture atlas.

    Returns the textured mesh and the texture image (H, W, 3) uint8 RGB.
    """
    import xatlas

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    print(f"[texture] running xatlas on {len(faces)} faces …")
    vmapping, new_faces, uvs = xatlas.parametrize(verts, faces)

    # Remap vertices/normals to match the xatlas output
    new_verts = verts[vmapping]
    new_normals = normals[vmapping]

    texture = np.zeros((texture_size, texture_size, 3), dtype=np.float64)
    weight_map = np.zeros((texture_size, texture_size, 1), dtype=np.float64)

    for name in cam_names:
        K = cams[name]["K"].astype(np.float64)
        R = cams[name]["R"].astype(np.float64)
        T = cams[name]["T"].astype(np.float64)
        img = images[name]
        H_img, W_img = img.shape[:2]

        cam_center = (-R.T @ T).ravel()

        # For each face, project its 3 vertices, sample colour, write to UV
        for fi in range(len(new_faces)):
            f = new_faces[fi]
            v3d = new_verts[f]    # (3, 3)
            n3d = new_normals[f]  # (3, 3)
            uv = uvs[f]          # (3, 2)  in [0,1]

            # Check visibility
            face_center = v3d.mean(axis=0)
            face_normal = n3d.mean(axis=0)
            face_normal /= np.linalg.norm(face_normal) + 1e-12
            view_dir = cam_center - face_center
            view_dir /= np.linalg.norm(view_dir) + 1e-12
            cos_a = np.dot(face_normal, view_dir)
            if cos_a < 0.05:
                continue

            pts_cam = (R @ v3d.T + T).T
            if (pts_cam[:, 2] <= 0).any():
                continue
            proj = (K @ pts_cam.T).T
            px = proj[:, 0] / pts_cam[:, 2]
            py = proj[:, 1] / pts_cam[:, 2]
            if (px < 0).any() or (px >= W_img).any() or (py < 0).any() or (py >= H_img).any():
                continue

            # Rasterise the UV triangle into the texture
            uv_px = uv.copy()
            uv_px[:, 0] *= texture_size - 1
            uv_px[:, 1] *= texture_size - 1
            uv_px[:, 1] = texture_size - 1 - uv_px[:, 1]  # flip V

            bbox_min = np.floor(uv_px.min(axis=0)).astype(int)
            bbox_max = np.ceil(uv_px.max(axis=0)).astype(int)
            bbox_min = np.clip(bbox_min, 0, texture_size - 1)
            bbox_max = np.clip(bbox_max, 0, texture_size - 1)

            for ty in range(bbox_min[1], bbox_max[1] + 1):
                for tx in range(bbox_min[0], bbox_max[0] + 1):
                    # Barycentric coordinates in UV space
                    p = np.array([tx, ty], dtype=np.float64)
                    v0 = uv_px[2] - uv_px[0]
                    v1 = uv_px[1] - uv_px[0]
                    v2 = p - uv_px[0]
                    d00 = v0 @ v0
                    d01 = v0 @ v1
                    d02 = v0 @ v2
                    d11 = v1 @ v1
                    d12 = v1 @ v2
                    denom = d00 * d11 - d01 * d01
                    if abs(denom) < 1e-10:
                        continue
                    inv_d = 1.0 / denom
                    u_b = (d11 * d02 - d01 * d12) * inv_d
                    v_b = (d00 * d12 - d01 * d02) * inv_d
                    w_b = 1.0 - u_b - v_b
                    if u_b < -0.01 or v_b < -0.01 or w_b < -0.01:
                        continue

                    # Interpolate image coordinates
                    img_x = w_b * px[0] + v_b * px[1] + u_b * px[2]
                    img_y = w_b * py[0] + v_b * py[1] + u_b * py[2]
                    ix = int(np.clip(np.round(img_x), 0, W_img - 1))
                    iy = int(np.clip(np.round(img_y), 0, H_img - 1))

                    colour_bgr = img[iy, ix].astype(np.float64)
                    colour_rgb = colour_bgr[::-1]

                    w = cos_a
                    texture[ty, tx] += colour_rgb * w
                    weight_map[ty, tx] += w

    valid = weight_map[..., 0] > 0
    texture[valid] /= weight_map[valid]
    texture = np.clip(texture, 0, 255).astype(np.uint8)

    # Build textured trimesh
    from PIL import Image
    tex_image = Image.fromarray(texture)

    uv_visual = trimesh.visual.TextureVisuals(uv=uvs, image=tex_image)
    mesh_out = trimesh.Trimesh(
        vertices=new_verts,
        faces=new_faces,
        visual=uv_visual,
        process=False,
    )

    print(f"[texture] baked {texture_size}x{texture_size} texture")
    return mesh_out, texture
