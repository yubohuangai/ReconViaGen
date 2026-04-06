#!/usr/bin/env python3
"""Classical multi-view stereo reconstruction.

Reads 11 calibrated RGB views (EasyMocap format) and produces a triangle
mesh via plane-sweep MVS depth estimation + Open3D TSDF fusion.

Example
-------
    python reconstruct_classical.py \\
        --data_root /path/to/data \\
        --device cuda:0

    Writes ``<data_root>/reconstruction/mesh.glb`` by default. **Foreground
    masks are required** under ``<data_root>/masks/<cam>/`` (see Motion-Capture
    ``generate_masks.py``).
"""

import argparse
import os
import time
from os.path import join


def parse_args():
    p = argparse.ArgumentParser(
        description="Classical MVS reconstruction from calibrated multi-view images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Foreground masks are required under <data_root>/<masks>/<cam>/ "
            "(default subdir name: masks). Override path with --masks."
        ),
    )

    # I/O
    p.add_argument("--data_root", required=True,
                    help="Directory containing images/<cam>/ + intri.yml + extri.yml")
    p.add_argument("--output_dir", default=None,
                    help="Where to write mesh and optional depth maps. "
                         "Default: <data_root>/reconstruction (not the repo cwd)")
    p.add_argument("--frame", type=int, default=0,
                    help="Frame index (selects {frame:06d}{ext} per camera)")
    p.add_argument("--ext", default=".jpg",
                    help="Image file extension")
    p.add_argument("--intri", default="intri.yml",
                    help="Intrinsics file name (relative to data_root)")
    p.add_argument("--extri", default="extri.yml",
                    help="Extrinsics file name (relative to data_root)")
    p.add_argument("--masks", default="masks",
                    help="Mask subdirectory under data_root (required; default: masks)")
    p.add_argument("--format", choices=["glb", "obj"], default="glb",
                    help="Output mesh format")

    # Plane-sweep MVS
    p.add_argument("--num_depths", type=int, default=192,
                    help="Number of depth hypotheses")
    p.add_argument("--num_sources", type=int, default=4,
                    help="Source views per reference view")
    p.add_argument("--window_size", type=int, default=7,
                    help="Cost-aggregation window size")
    p.add_argument("--depth_min", type=float, default=None,
                    help="Override minimum depth (world units)")
    p.add_argument("--depth_max", type=float, default=None,
                    help="Override maximum depth (world units)")
    p.add_argument("--confidence_threshold", type=float, default=0.3,
                    help="Discard depth pixels below this confidence")

    # TSDF
    p.add_argument("--tsdf_voxel_length", type=float, default=None,
                    help="TSDF voxel size in world units (auto if omitted)")
    p.add_argument("--sdf_trunc_factor", type=float, default=3.0,
                    help="SDF truncation = factor * voxel_length")
    p.add_argument("--no_clean", action="store_true",
                    help="Skip point-cloud cleaning after TSDF extraction")

    # Texturing
    p.add_argument("--texture", action="store_true",
                    help="Bake a UV texture map (slow; default: vertex colours only)")
    p.add_argument("--texture_size", type=int, default=2048,
                    help="Texture atlas resolution")
    p.add_argument("--refine_colors", action="store_true",
                    help="Re-project vertex colours from calibrated views "
                         "(improves TSDF default colours)")

    # Misc
    p.add_argument("--device", default="cuda:0",
                    help="Torch device for plane-sweep MVS")
    p.add_argument("--save_depths", action="store_true",
                    help="Save per-view depth maps as .npy + 16-bit PNG")
    p.add_argument("--undistort", action="store_true", default=True,
                    help="Undistort images before processing")
    p.add_argument("--no_undistort", action="store_true",
                    help="Skip image undistortion")

    return p.parse_args()


def main():
    args = parse_args()
    if args.no_undistort:
        args.undistort = False

    import cv2
    import numpy as np

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else join(args.data_root, "reconstruction")
    )
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1.  Load cameras
    # ------------------------------------------------------------------
    from classical_mvs.camera_io import (
        read_cameras, load_images, load_masks, undistort_image,
    )

    intri_path = join(args.data_root, args.intri)
    extri_path = join(args.data_root, args.extri)
    print(f"[main] Loading cameras from {intri_path}, {extri_path}")

    cams, cam_names = read_cameras(intri_path, extri_path)
    print(f"[main] {len(cam_names)} cameras: {cam_names}")
    print(f"[main] Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # 2.  Load images and required foreground masks
    # ------------------------------------------------------------------
    print("[main] Loading images …")
    images = load_images(args.data_root, cam_names, frame=args.frame, ext=args.ext)

    mask_root = join(args.data_root, args.masks)
    print(f"[main] Loading masks from {mask_root}/ …")
    masks = load_masks(
        args.data_root,
        cam_names,
        mask_dir=args.masks,
        frame=args.frame,
        required=True,
    )
    print(f"[main] Loaded masks for {len(cam_names)} views")

    # Undistort
    if args.undistort:
        for name in cam_names:
            images[name] = undistort_image(
                images[name], cams[name]["K"], cams[name]["dist"],
            )

    # Update H/W from actual image size (calibration may not have it)
    for name in cam_names:
        h, w = images[name].shape[:2]
        if cams[name]["H"] <= 0:
            cams[name]["H"] = h
        if cams[name]["W"] <= 0:
            cams[name]["W"] = w

    # ------------------------------------------------------------------
    # 3.  Plane-sweep MVS depth estimation
    # ------------------------------------------------------------------
    from classical_mvs.plane_sweep import estimate_depth_maps

    depth_range = None
    if args.depth_min is not None and args.depth_max is not None:
        depth_range = (args.depth_min, args.depth_max)

    print("[main] Running plane-sweep MVS …")
    depth_maps, conf_maps = estimate_depth_maps(
        images, cam_names, cams,
        num_depths=args.num_depths,
        num_sources=args.num_sources,
        window_size=args.window_size,
        depth_range=depth_range,
        masks=masks,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )

    if args.save_depths:
        depth_dir = join(output_dir, "depth_maps")
        os.makedirs(depth_dir, exist_ok=True)
        for name in cam_names:
            np.save(join(depth_dir, f"{name}.npy"), depth_maps[name])
            # 16-bit PNG (scale to millimetres)
            dm_mm = (depth_maps[name] * 1000).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(join(depth_dir, f"{name}.png"), dm_mm)
            np.save(join(depth_dir, f"{name}_conf.npy"), conf_maps[name])
        print(f"[main] Saved depth maps to {depth_dir}/")

    # ------------------------------------------------------------------
    # 4.  TSDF fusion
    # ------------------------------------------------------------------
    from classical_mvs.tsdf_fusion import fuse_tsdf

    print("[main] Running TSDF fusion …")
    mesh = fuse_tsdf(
        images, depth_maps, cams, cam_names,
        voxel_length=args.tsdf_voxel_length,
        sdf_trunc_factor=args.sdf_trunc_factor,
        masks=masks,
        clean=not args.no_clean,
    )

    # ------------------------------------------------------------------
    # 5.  Vertex colour refinement (optional)
    # ------------------------------------------------------------------
    if args.refine_colors:
        from classical_mvs.texturing import refine_vertex_colors

        print("[main] Refining vertex colours …")
        mesh = refine_vertex_colors(mesh, images, cams, cam_names, masks=masks)

    # ------------------------------------------------------------------
    # 6.  Texture bake (optional)
    # ------------------------------------------------------------------
    if args.texture:
        from classical_mvs.texturing import bake_texture

        print("[main] Baking texture atlas …")
        mesh, tex_img = bake_texture(
            mesh, images, cams, cam_names,
            texture_size=args.texture_size, masks=masks,
        )
        # Save the texture image separately for OBJ format
        tex_path = join(output_dir, "texture.png")
        cv2.imwrite(tex_path, tex_img[:, :, ::-1])
        print(f"[main] Saved texture to {tex_path}")

    # ------------------------------------------------------------------
    # 7.  Export
    # ------------------------------------------------------------------
    if args.format == "glb":
        out_path = join(output_dir, "mesh.glb")
        mesh.export(out_path)
    else:
        out_path = join(output_dir, "mesh.obj")
        mesh.export(out_path)

    elapsed = time.time() - t0
    print(f"[main] Done in {elapsed:.1f}s — wrote {out_path}")
    print(f"[main]   {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")


if __name__ == "__main__":
    main()
