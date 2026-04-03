#!/usr/bin/env python3
"""
Multi-view 3D reconstruction without Gradio.

Loads N view images (e.g. 11), runs TrellisVGGTTo3DPipeline, writes mesh as GLB
and optionally Gaussian splats as PLY.

Example:
  python infer_multiview.py --images view_00.png view_01.png ... view_10.png --out_dir ./out_run1

  python infer_multiview.py --image_dir ./my_views --out_dir ./out_run1
  # (all .png/.jpg/.jpeg/.webp in the folder, sorted by filename)

  python infer_multiview.py --views_root /mnt/yubo/obj/cube/images --out_dir ./out_cube
  # expects .../images/01/000000.jpg, .../02/000000.jpg, ... (one frame per view subfolder)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("SPCONV_ALGO", "native")
# Reduces allocator fragmentation when peak VRAM is tight (e.g. 11-view VGGT on 11GB).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image

from trellis.pipelines import TrellisVGGTTo3DPipeline
from trellis.utils import postprocessing_utils


def _natural_sort_key(name: str):
    """Sort '1'..'11' numerically when names are numeric; else lexicographic chunks."""
    parts = re.split(r"(\d+)", name)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        elif p:
            key.append(p.lower())
    return key


def collect_paths_from_dir(image_dir: str) -> list[str]:
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {image_dir}")
    exts = {".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP"}
    files = sorted(f for f in root.iterdir() if f.is_file() and f.suffix in exts)
    return [str(f) for f in files]


def collect_paths_from_view_subdirs(views_root: str, frame_name: str) -> list[str]:
    """
    One image per immediate subfolder of ``views_root``, e.g.::

        views_root/01/000000.jpg
        views_root/02/000000.jpg
        ...

    Subfolders are ordered by *natural* sort on the folder name (01, 02, …, 11).
    """
    root = Path(views_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {views_root}")
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    subdirs.sort(key=lambda p: _natural_sort_key(p.name))
    paths: list[str] = []
    for d in subdirs:
        f = d / frame_name
        if not f.is_file():
            raise FileNotFoundError(
                f"Expected file not found: {f}\n"
                f"  (set --frame_name if your frame file name differs)"
            )
        paths.append(str(f.resolve()))
    return paths


def offload_to_cpu_for_low_vram(pipeline: TrellisVGGTTo3DPipeline) -> None:
    """
    from_pretrained loads VGGT, BiRefNet, DreamSim, and Trellis weights onto GPU.
    For 11-view VGGT, that leaves almost no room before aggregator forward.
    Move everything to CPU first; keep BiRefNet on GPU only during preprocess, then CPU again before run().
    """
    pipeline.VGGT_model.cpu()
    if getattr(pipeline, "dreamsim_model", None) is not None:
        pipeline.dreamsim_model.cpu()
    for model in pipeline.models.values():
        model.cpu()
    pipeline.birefnet_model.cpu()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def load_and_preprocess(
    paths: list[str],
    pipeline: TrellisVGGTTo3DPipeline,
) -> list[Image.Image]:
    images = []
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        images.append(Image.open(p))
    return [pipeline.preprocess_image(im) for im in images]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ReconViaGen multi-view inference (code-only, no Gradio)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--images",
        nargs="+",
        metavar="PATH",
        help="Paths to view images in camera / capture order (e.g. 11 files).",
    )
    src.add_argument(
        "--image_dir",
        type=str,
        help="Directory of views; uses *.png *.jpg *.jpeg *.webp sorted by name.",
    )
    src.add_argument(
        "--views_root",
        type=str,
        metavar="DIR",
        help=(
            "Root with one subfolder per view, each containing the same frame file name, e.g. "
            "DIR/01/000000.jpg, DIR/02/000000.jpg. Subfolders ordered by natural sort."
        ),
    )
    p.add_argument(
        "--frame_name",
        type=str,
        default="000000.jpg",
        help="With --views_root: image file name inside each view subfolder (default: 000000.jpg).",
    )

    p.add_argument(
        "--pretrained",
        type=str,
        default="Stable-X/trellis-vggt-v0-2",
        help="Hugging Face model id for TrellisVGGTTo3DPipeline.",
    )
    p.add_argument("--out_dir", type=str, default="outputs/infer_multiview", help="Output directory.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mode",
        type=str,
        choices=("multidiffusion", "stochastic"),
        default="multidiffusion",
        help="Multi-view fusion mode (same as Gradio default).",
    )
    p.add_argument(
        "--no_low_vram",
        action="store_true",
        help="Keep major models on GPU (faster, needs more VRAM). Default is low-VRAM mode like app.py.",
    )
    p.add_argument(
        "--cuda_device",
        type=int,
        default=None,
        metavar="N",
        help="CUDA device index (e.g. 1 on a multi-GPU box). Default: current device / 0.",
    )

    # Stage 1 (sparse structure) — defaults match app.py sliders
    p.add_argument("--ss_steps", type=int, default=30)
    p.add_argument("--ss_cfg", type=float, default=7.5)
    p.add_argument("--ss_guidance_rescale", type=float, default=0.7)
    p.add_argument("--ss_rescale_t", type=float, default=5.0)

    # Stage 2 (SLAT)
    p.add_argument("--slat_steps", type=int, default=12)
    p.add_argument("--slat_cfg", type=float, default=7.5)
    p.add_argument("--slat_guidance_rescale", type=float, default=0.5)
    p.add_argument("--slat_rescale_t", type=float, default=3.0)

    # Export
    p.add_argument("--mesh_simplify", type=float, default=0.95)
    p.add_argument("--texture_size", type=int, default=1024, choices=(512, 1024, 2048))
    p.add_argument("--save_ply", action="store_true", help="Also save Gaussian PLY (can be large).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    low_vram = not args.no_low_vram

    if args.images:
        paths = list(args.images)
    elif args.image_dir:
        paths = collect_paths_from_dir(args.image_dir)
    else:
        paths = collect_paths_from_view_subdirs(args.views_root, args.frame_name)

    if len(paths) == 0:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[infer] {len(paths)} views (in order):")
    for i, pth in enumerate(paths):
        print(f"  {i:02d}  {pth}")
    if len(paths) != 11:
        print(f"[infer] note: {len(paths)} views loaded; use subfolder names to control order.")

    if not torch.cuda.is_available():
        print("CUDA is required for this pipeline.", file=sys.stderr)
        sys.exit(1)
    if args.cuda_device is not None:
        torch.cuda.set_device(args.cuda_device)
    device = torch.device(f"cuda:{args.cuda_device}" if args.cuda_device is not None else "cuda")

    print(f"[infer] loading pipeline: {args.pretrained}")
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
    pipeline._device = device
    pipeline.low_vram = low_vram

    if low_vram:
        print("[infer] low VRAM: moving Trellis / VGGT / DreamSim / BiRefNet off GPU, then BiRefNet only for preprocess")
        offload_to_cpu_for_low_vram(pipeline)
        pipeline.birefnet_model.to(device)
    else:
        for model in pipeline.models.values():
            model.to(device)
        pipeline.VGGT_model.to(device)
        if getattr(pipeline, "dreamsim_model", None) is not None:
            pipeline.dreamsim_model.to(device)
        pipeline.birefnet_model.to(device)

    print("[infer] preprocessing (BiRefNet / crop, same as demo)")
    image_files = load_and_preprocess(paths, pipeline)

    if low_vram:
        pipeline.birefnet_model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("[infer] BiRefNet off GPU; freed memory for VGGT (11 views)")

    sparse_structure_sampler_params = {
        "steps": args.ss_steps,
        "cfg_strength": args.ss_cfg,
        "cfg_interval": [0.6, 1.0],
        "guidance_rescale": args.ss_guidance_rescale,
        "rescale_t": args.ss_rescale_t,
    }
    slat_sampler_params = {
        "steps": args.slat_steps,
        "cfg_strength": args.slat_cfg,
        "cfg_interval": [0.6, 1.0],
        "guidance_rescale": args.slat_guidance_rescale,
        "rescale_t": args.slat_rescale_t,
    }

    print("[infer] sampling …")
    outputs, _, _ = pipeline.run(
        image=image_files,
        seed=args.seed,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params=sparse_structure_sampler_params,
        slat_sampler_params=slat_sampler_params,
        mode=args.mode,
    )

    gs = outputs["gaussian"][0]
    mesh = outputs["mesh"][0]
    del outputs
    torch.cuda.empty_cache()

    glb_path = out_dir / "mesh.glb"
    print(f"[infer] exporting GLB -> {glb_path}")
    glb = postprocessing_utils.to_glb(
        gs, mesh, simplify=args.mesh_simplify, texture_size=args.texture_size, verbose=True
    )
    glb.export(str(glb_path))

    if args.save_ply:
        ply_path = out_dir / "gaussian.ply"
        print(f"[infer] exporting PLY -> {ply_path}")
        gs.save_ply(str(ply_path))

    print("[infer] done.")


if __name__ == "__main__":
    main()
