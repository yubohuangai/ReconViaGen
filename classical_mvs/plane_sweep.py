"""Plane-sweep multi-view stereo depth estimation (PyTorch / GPU).

Classical algorithm — no learned priors.  For each *reference* view the
module sweeps D fronto-parallel depth hypotheses, warps every selected
*source* view to the reference via per-plane homographies, computes a
photo-consistency cost (variance), and picks the winner-take-all depth.
Post-filters include geometric consistency across views and confidence
thresholding.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _camera_center(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """World-space camera centre:  C = -R^T @ T."""
    return (-R.T @ T).ravel()


def _select_source_views(
    ref_idx: int,
    centers: np.ndarray,
    num_sources: int,
) -> List[int]:
    """Pick *num_sources* views with the best triangulation angle.

    A good source has a baseline that subtends a reasonable angle at the
    scene centre (≈ centroid of all cameras).  Very small angles give poor
    depth resolution; very large angles hurt photo-consistency.
    """
    scene_center = centers.mean(axis=0)
    ref_dir = centers[ref_idx] - scene_center
    ref_dir /= np.linalg.norm(ref_dir) + 1e-12

    scores = []
    for i, c in enumerate(centers):
        if i == ref_idx:
            scores.append(-1.0)
            continue
        d = c - scene_center
        d /= np.linalg.norm(d) + 1e-12
        # Favour angles ≈ 10°–60° — penalise <5° and >90°
        cos_a = np.clip(np.dot(ref_dir, d), -1, 1)
        angle = np.degrees(np.arccos(cos_a))
        score = np.exp(-((angle - 30) ** 2) / (2 * 25 ** 2))
        scores.append(score)

    order = np.argsort(scores)[::-1]
    return [int(i) for i in order[:num_sources]]


def _auto_depth_range(
    centers: np.ndarray,
    margin: float = 0.5,
) -> Tuple[float, float]:
    """Heuristic depth range from camera positions.

    Returns (depth_min, depth_max) in world units along any view's
    optical axis.  The object is assumed to be roughly at the centroid
    of the camera arrangement.
    """
    scene_center = centers.mean(axis=0)
    dists = np.linalg.norm(centers - scene_center, axis=1)
    d_min = max(dists.min() * margin, 1e-4)
    d_max = dists.max() * (1.0 + margin)
    return float(d_min), float(d_max)


# ---------------------------------------------------------------------------
# Core plane-sweep
# ---------------------------------------------------------------------------

def _build_homography(
    K_ref: torch.Tensor,       # (3,3)
    K_src: torch.Tensor,       # (3,3)
    R_ref: torch.Tensor,       # (3,3)
    T_ref: torch.Tensor,       # (3,1)
    R_src: torch.Tensor,       # (3,3)
    T_src: torch.Tensor,       # (3,1)
    depth: float,
) -> torch.Tensor:
    """Homography that warps *src* image to *ref* at a fronto-parallel plane
    at distance *depth* from the reference camera.

    H = K_ref @ (R_rel - t_rel @ n^T / d) @ inv(K_src)

    where n = [0,0,1]^T (ref optical axis), d = depth, and
    R_rel, t_rel convert from src-camera to ref-camera frame.
    """
    # Relative pose: src -> ref
    R_rel = R_ref @ R_src.T               # (3,3)
    t_rel = T_ref - R_rel @ T_src         # (3,1)

    n = torch.tensor([[0.0], [0.0], [1.0]], device=K_ref.device, dtype=K_ref.dtype)
    H = K_ref @ (R_rel - t_rel @ n.T / depth) @ torch.inverse(K_src)
    return H


def _warp_image(
    src: torch.Tensor,      # (1,C,H,W)
    H_matrix: torch.Tensor, # (3,3)
    height: int,
    width: int,
) -> torch.Tensor:
    """Warp *src* to the reference view via homography *H_matrix*."""
    # Build a grid of ref-pixel coordinates
    ys, xs = torch.meshgrid(
        torch.arange(height, device=src.device, dtype=src.dtype),
        torch.arange(width, device=src.device, dtype=src.dtype),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    coords = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3)  # (H*W, 3)

    # Map ref pixels -> src pixels through H^{-1}
    H_inv = torch.inverse(H_matrix)
    src_coords = (H_inv @ coords.T).T  # (H*W, 3)
    src_coords = src_coords[:, :2] / (src_coords[:, 2:3] + 1e-8)

    # Normalise to [-1, 1] for grid_sample
    src_coords[..., 0] = 2.0 * src_coords[..., 0] / (width - 1) - 1.0
    src_coords[..., 1] = 2.0 * src_coords[..., 1] / (height - 1) - 1.0
    grid = src_coords.reshape(1, height, width, 2)

    warped = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros",
                           align_corners=True)
    return warped  # (1,C,H,W)


def _compute_cost_volume(
    ref_img: torch.Tensor,          # (1,3,H,W)  float [0,1]
    src_imgs: List[torch.Tensor],   # each (1,3,H,W)
    K_ref: torch.Tensor,
    Ks_src: List[torch.Tensor],
    R_ref: torch.Tensor,
    T_ref: torch.Tensor,
    Rs_src: List[torch.Tensor],
    Ts_src: List[torch.Tensor],
    depths: torch.Tensor,           # (D,)
    window_size: int = 7,
    mask_ref: Optional[torch.Tensor] = None,  # (1,1,H,W) bool / float
) -> torch.Tensor:
    """Return cost volume (D, H, W) — lower is better (photo-inconsistency)."""
    D = depths.shape[0]
    _B, _C, H, W = ref_img.shape
    device = ref_img.device
    n_src = len(src_imgs)

    pad = window_size // 2
    avg_kernel = torch.ones(1, 1, window_size, window_size, device=device) / (window_size ** 2)

    cost_volume = torch.zeros(D, H, W, device=device)

    for di, d in enumerate(depths):
        # Accumulate variance across source views
        sum_c = torch.zeros(1, 3, H, W, device=device)
        sum_c2 = torch.zeros(1, 3, H, W, device=device)
        count = torch.zeros(1, 1, H, W, device=device)

        for si in range(n_src):
            Hm = _build_homography(
                K_ref, Ks_src[si], R_ref, T_ref,
                Rs_src[si], Ts_src[si], float(d),
            )
            warped = _warp_image(src_imgs[si], Hm, H, W)  # (1,3,H,W)
            # Validity mask: warped pixels that are non-zero
            valid = (warped.abs().sum(dim=1, keepdim=True) > 1e-6).float()
            sum_c += warped * valid
            sum_c2 += (warped ** 2) * valid
            count += valid

        count = count.clamp(min=1)
        mean = sum_c / count
        variance = (sum_c2 / count - mean ** 2).mean(dim=1, keepdim=True)  # (1,1,H,W)
        variance = variance.clamp(min=0)

        # Window-aggregate the variance for robustness
        var_padded = F.pad(variance, [pad] * 4, mode="reflect")
        var_smooth = F.conv2d(var_padded, avg_kernel)  # (1,1,H,W)

        cost_volume[di] = var_smooth.squeeze()

    if mask_ref is not None:
        cost_volume[:, ~mask_ref.squeeze().bool()] = 1e6

    return cost_volume


# ---------------------------------------------------------------------------
# Geometric consistency filter
# ---------------------------------------------------------------------------

def _geometric_consistency(
    depth_maps: Dict[int, np.ndarray],
    Ks: List[np.ndarray],
    Rs: List[np.ndarray],
    Ts: List[np.ndarray],
    source_indices: Dict[int, List[int]],
    thresh_px: float = 1.0,
    thresh_depth_rel: float = 0.01,
) -> Dict[int, np.ndarray]:
    """Filter depth maps by reprojection into neighbouring views.

    For every pixel in the reference view, project to 3-D, then into each
    source.  Keep the pixel only if at least one source agrees (depth &
    reprojection within thresholds).  Returns confidence maps (0-1).
    """
    confidence: Dict[int, np.ndarray] = {}
    for ref_idx, depth_ref in depth_maps.items():
        H, W = depth_ref.shape
        K_r, R_r, T_r = Ks[ref_idx], Rs[ref_idx], Ts[ref_idx]
        invK = np.linalg.inv(K_r)

        ys, xs = np.mgrid[:H, :W].astype(np.float64)
        ones = np.ones_like(xs)
        pixels = np.stack([xs, ys, ones], axis=-1)  # (H,W,3)

        rays = (invK @ pixels.reshape(-1, 3).T).T.reshape(H, W, 3)
        pts_cam = rays * depth_ref[..., None]
        pts_world = (R_r.T @ (pts_cam.reshape(-1, 3).T - T_r)).T.reshape(H, W, 3)

        votes = np.zeros((H, W), dtype=np.float32)
        for src_idx in source_indices.get(ref_idx, []):
            if src_idx not in depth_maps:
                continue
            K_s, R_s, T_s = Ks[src_idx], Rs[src_idx], Ts[src_idx]
            depth_src = depth_maps[src_idx]

            pts_src_cam = (R_s @ pts_world.reshape(-1, 3).T + T_s).T.reshape(H, W, 3)
            z_src = pts_src_cam[..., 2]

            proj = (K_s @ pts_src_cam.reshape(-1, 3).T).T.reshape(H, W, 3)
            u = proj[..., 0] / (proj[..., 2] + 1e-8)
            v = proj[..., 1] / (proj[..., 2] + 1e-8)

            H_s, W_s = depth_src.shape
            in_bounds = (u >= 0) & (u < W_s - 1) & (v >= 0) & (v < H_s - 1) & (z_src > 0)
            ui = np.clip(np.round(u).astype(int), 0, W_s - 1)
            vi = np.clip(np.round(v).astype(int), 0, H_s - 1)
            depth_sampled = depth_src[vi, ui]

            reproj_ok = np.sqrt((u - np.round(u)) ** 2 + (v - np.round(v)) ** 2) < thresh_px
            depth_ok = np.abs(z_src - depth_sampled) / (depth_sampled + 1e-8) < thresh_depth_rel
            votes += (in_bounds & reproj_ok & depth_ok).astype(np.float32)

        confidence[ref_idx] = votes / max(1, len(source_indices.get(ref_idx, [])))

    return confidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_depth_maps(
    images: Dict[str, np.ndarray],
    cam_names: List[str],
    cams: Dict[str, dict],
    *,
    num_depths: int = 192,
    num_sources: int = 4,
    window_size: int = 7,
    depth_range: Optional[Tuple[float, float]] = None,
    masks: Optional[Dict[str, np.ndarray]] = None,
    device: str = "cuda:0",
    geo_consistency: bool = True,
    confidence_threshold: float = 0.3,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Estimate per-view depth maps via plane-sweep stereo.

    Parameters
    ----------
    images : dict  cam_name -> BGR uint8 (H,W,3)
    cam_names : ordered camera name list
    cams : dict   cam_name -> {K, R, T, ...}
    num_depths : number of depth hypotheses
    num_sources : source views per reference
    window_size : aggregation window
    depth_range : (min, max) in world units; auto if None
    masks : optional foreground masks (>0 = foreground)
    device : torch device string
    geo_consistency : enable geometric consistency filtering
    confidence_threshold : discard depths with conf below this

    Returns
    -------
    depth_maps : dict  cam_name -> (H, W) float depth in camera coords
    conf_maps  : dict  cam_name -> (H, W) float [0, 1]
    """
    dev = torch.device(device)
    N = len(cam_names)

    # Numpy camera arrays (index-addressable)
    Ks = [cams[c]["K"].astype(np.float64) for c in cam_names]
    Rs = [cams[c]["R"].astype(np.float64) for c in cam_names]
    Ts = [cams[c]["T"].astype(np.float64) for c in cam_names]
    centers = np.array([_camera_center(R, T) for R, T in zip(Rs, Ts)])

    if depth_range is None:
        d_min, d_max = _auto_depth_range(centers)
    else:
        d_min, d_max = depth_range
    print(f"[plane_sweep] depth range: [{d_min:.4f}, {d_max:.4f}]")

    # Inverse-depth sampling (denser near the cameras)
    inv_depths = torch.linspace(1.0 / d_max, 1.0 / d_min, num_depths, device=dev)
    depths = 1.0 / inv_depths  # (D,) large -> small

    # Prepare torch tensors for images
    img_tensors: List[torch.Tensor] = []
    mask_tensors: List[Optional[torch.Tensor]] = []
    for c in cam_names:
        img = images[c]
        t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensors.append(t.to(dev))
        if masks is not None and masks.get(c) is not None:
            m = torch.from_numpy(masks[c]).float().unsqueeze(0).unsqueeze(0) / 255.0
            mask_tensors.append((m > 0.5).to(dev))
        else:
            mask_tensors.append(None)

    K_ts = [torch.from_numpy(K).float().to(dev) for K in Ks]
    R_ts = [torch.from_numpy(R).float().to(dev) for R in Rs]
    T_ts = [torch.from_numpy(T).float().to(dev) for T in Ts]

    source_map: Dict[int, List[int]] = {}
    raw_depth_maps: Dict[int, np.ndarray] = {}
    depth_results: Dict[str, np.ndarray] = {}

    for ref_i in range(N):
        src_indices = _select_source_views(ref_i, centers, num_sources)
        source_map[ref_i] = src_indices
        ref_name = cam_names[ref_i]
        H, W = images[ref_name].shape[:2]

        print(f"[plane_sweep] view {ref_name} ({ref_i+1}/{N}), "
              f"sources={[cam_names[s] for s in src_indices]}")

        with torch.no_grad():
            cost = _compute_cost_volume(
                ref_img=img_tensors[ref_i],
                src_imgs=[img_tensors[si] for si in src_indices],
                K_ref=K_ts[ref_i],
                Ks_src=[K_ts[si] for si in src_indices],
                R_ref=R_ts[ref_i],
                T_ref=T_ts[ref_i],
                Rs_src=[R_ts[si] for si in src_indices],
                Ts_src=[T_ts[si] for si in src_indices],
                depths=depths,
                window_size=window_size,
                mask_ref=mask_tensors[ref_i],
            )
            best_idx = cost.argmin(dim=0)  # (H, W)
            depth_map = depths[best_idx.reshape(-1)].reshape(H, W).cpu().numpy()

        raw_depth_maps[ref_i] = depth_map

    # Geometric consistency
    if geo_consistency:
        print("[plane_sweep] geometric consistency filtering …")
        conf_maps_idx = _geometric_consistency(
            raw_depth_maps, Ks, Rs, Ts, source_map,
        )
    else:
        conf_maps_idx = {i: np.ones_like(d) for i, d in raw_depth_maps.items()}

    # Package results keyed by camera name
    conf_results: Dict[str, np.ndarray] = {}
    for i, name in enumerate(cam_names):
        dm = raw_depth_maps[i]
        conf = conf_maps_idx.get(i, np.ones_like(dm))
        dm[conf < confidence_threshold] = 0.0
        depth_results[name] = dm
        conf_results[name] = conf

    return depth_results, conf_results
