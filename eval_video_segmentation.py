#!/usr/bin/env python3
"""DAVIS 2017 video object segmentation with a frozen ResNet-50.

Label propagation matches
https://github.com/facebookresearch/dino/blob/main/eval_video_segmentation.py
(Apache-2.0; the propagation itself follows https://github.com/Liusifei/UVC).
ViT patch tokens are replaced by the ResNet-50 ``layer4`` map. ``--stride`` is
that map's downsampling factor, the DINO ``patch_size``:

* 32 — standard ResNet-50
* 16 — ``layer4`` stride replaced by dilation 2
* 8 — ``layer3`` and ``layer4`` strides replaced by dilation

Weights are unchanged by dilation; only the sampling grid changes. Checkpoints
use the same backbone extraction as ``extract_backbone.py`` (SimCLR / SimLAP /
MoCo projectors are dropped).

Frames are resized as in the DINO script (shorter side 480, longer side snapped
down to a multiple of 64) and normalized with ImageNet mean/std. The script
writes indexed PNG masks. When every annotation frame is present it also
reports region similarity J and contour accuracy F (DAVIS J&F).

DAVIS layout (2017 val):

```
DAVIS/
  ImageSets/2017/val.txt
  JPEGImages/480p/<video>/*.jpg
  Annotations/480p/<video>/*.png
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet50

from extract_backbone import extract_backbone, get_state_dict, load_checkpoint

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SHORTER = 480
MULTIPLE = 64
TEMPERATURE = 0.1
STRIDES = (8, 16, 32)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional
    def tqdm(iterable, **_kwargs):
        return iterable

_NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")
_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - requirements.txt includes scipy
    ndimage = None


def make_resnet50(stride: int) -> nn.Module:
    """Torchvision ResNet-50 whose ``layer4`` map is ``stride`` pixels per cell."""
    if stride not in STRIDES:
        raise ValueError(f"stride must be one of {STRIDES}")
    # torchvision flags are (layer2, layer3, layer4): replace the stride-2
    # downsample with dilation so the pretrained kernels still load.
    dilate = {
        8: [False, True, True],
        16: [False, False, True],
        32: [False, False, False],
    }[stride]
    kwargs = {"replace_stride_with_dilation": dilate}
    try:
        model = resnet50(weights=None, **kwargs)
    except TypeError:
        model = resnet50(pretrained=False, **kwargs)
    model.feature_stride = stride  # type: ignore[attr-defined]
    return model


def load_resnet50(path: Path, stride: int) -> nn.Module:
    """Load a frozen ResNet-50. ``path`` is a training checkpoint or a backbone."""
    model = make_resnet50(stride)
    model.fc = nn.Identity()
    state = extract_backbone(get_state_dict(load_checkpoint(path)))
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"ResNet-50 checkpoint {path} does not match: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def forward_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """``(N, 2048, H/stride, W/stride)`` map, before global pooling."""
    x = model.relu(model.bn1(model.conv1(images)))
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    return model.layer4(x)


def extract_feature(
    model: nn.Module, frame: torch.Tensor, device: torch.device, return_hw: bool = False
):
    """One frame -> ``(h*w, dim)`` tokens, the ViT patch-token layout."""
    feat = forward_features(model, frame.unsqueeze(0).to(device, non_blocking=True))
    _, dim, height, width = feat.shape
    stride = int(model.feature_stride)
    if frame.shape[-2] != height * stride or frame.shape[-1] != width * stride:
        raise RuntimeError(
            f"feature map {(height, width)} times stride {stride} "
            f"!= frame {tuple(frame.shape[-2:])}"
        )
    tokens = feat[0].permute(1, 2, 0).reshape(-1, dim)
    if return_hw:
        return tokens, height, width
    return tokens


def restrict_neighborhood(height: int, width: int, radius: int, device: torch.device) -> torch.Tensor:
    """``(h*w, h*w)`` mask; a query may only attend inside a Chebyshev ball."""
    yy = torch.arange(height, device=device)
    xx = torch.arange(width, device=device)
    near_y = (yy[:, None] - yy[None, :]).abs() <= radius
    near_x = (xx[:, None] - xx[None, :]).abs() <= radius
    # mask[i, j, p, q] = 1 when |i-p|<=r and |j-q|<=r. Same indexing as the
    # quadruple loop in the DINO script, without the Python loops.
    return (near_y[:, None, :, None] & near_x[None, :, None, :]).reshape(height * width, height * width)


def label_propagation(
    model: nn.Module,
    frame: torch.Tensor,
    source_feats: list[torch.Tensor],
    source_segs: list[torch.Tensor],
    *,
    topk: int,
    radius: int | None,
    neighborhood: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Propagate source label maps onto ``frame``.

    ``source_feats`` are ``(dim, h*w)``. ``source_segs`` are ``(1, C, h, w)``.
    Returns the soft target map ``(1, C, h, w)``, the target tokens as
    ``(dim, h*w)``, and the cached neighborhood mask.
    """
    feat_tar, height, width = extract_feature(model, frame, device, return_hw=True)
    stored = feat_tar.transpose(0, 1).contiguous()  # dim x h*w
    ncontext = len(source_feats)
    sources = torch.stack(source_feats)  # n x dim x h*w
    query = F.normalize(feat_tar, dim=1, p=2).unsqueeze(0).expand(ncontext, -1, -1)
    keys = F.normalize(sources, dim=1, p=2)
    aff = torch.exp(torch.bmm(query, keys) / TEMPERATURE)  # n x hw_tar x hw_src

    if radius is not None:
        if neighborhood is None or neighborhood.shape[-1] != height * width:
            neighborhood = restrict_neighborhood(height, width, radius, aff.device)
        aff = aff * neighborhood.to(dtype=aff.dtype)

    # Columns are queries. Keep the top-k source locations, then make them sum to 1.
    flat = aff.transpose(2, 1).reshape(-1, height * width)  # (n*hw_src) x hw_tar
    k = min(topk, flat.shape[0])
    if k < 1:
        raise RuntimeError("no source locations to propagate from")
    smallest_kept, _ = torch.topk(flat, k=k, dim=0).values.min(dim=0)
    flat = flat.masked_fill(flat < smallest_kept, 0)
    flat = flat / flat.sum(dim=0, keepdim=True).clamp_min(1e-12)

    masks = torch.cat([seg.to(flat.device) for seg in source_segs])  # n x C x h x w
    classes = masks.shape[1]
    masks = masks.reshape(ncontext, classes, -1).transpose(2, 1).reshape(-1, classes).transpose(0, 1)
    seg_tar = torch.mm(masks, flat).reshape(1, classes, height, width)
    return seg_tar, stored, neighborhood


def scaled_size(height: int, width: int, shorter: int = SHORTER, multiple: int = MULTIPLE) -> tuple[int, int]:
    """DINO frame resize: shorter side fixed, longer side snapped down to ``multiple``."""
    if height > width:
        tw = float(shorter)
        th = (tw * height) / width
        th = int((th // multiple) * multiple)
        tw = int(tw)
    else:
        th = float(shorter)
        tw = (th * width) / height
        tw = int((tw // multiple) * multiple)
        th = int(th)
    if th < 1 or tw < 1:
        raise ValueError(f"scaled size {(th, tw)} is empty")
    return th, tw


def color_normalize(frame: torch.Tensor) -> torch.Tensor:
    mean = frame.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = frame.new_tensor(IMAGENET_STD)[:, None, None]
    return (frame - mean) / std


def read_frame(path: Path, shorter: int = SHORTER) -> tuple[torch.Tensor, int, int]:
    image = Image.open(path).convert("RGB")
    ori_w, ori_h = image.size
    th, tw = scaled_size(ori_h, ori_w, shorter=shorter)
    image = image.resize((tw, th), _BILINEAR)
    frame = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
    return color_normalize(frame), ori_h, ori_w


def to_one_hot(labels: torch.Tensor) -> torch.Tensor:
    """``(1, H, W)`` integer labels -> ``(1, C, H, W)`` float one-hot."""
    _, height, width = labels.shape
    flat = labels.reshape(-1).long()
    if int(flat.min()) < 0:
        raise ValueError("negative labels")
    classes = int(flat.max()) + 1
    eye = torch.zeros(flat.numel(), classes)
    eye.scatter_(1, flat[:, None], 1.0)
    return eye.view(height, width, classes).permute(2, 0, 1).unsqueeze(0)


def read_seg(path: Path, stride: int, shorter: int = SHORTER) -> tuple[torch.Tensor, np.ndarray]:
    seg = Image.open(path)
    original = np.array(seg)
    if original.ndim != 2:
        raise ValueError(f"{path} is not a 2D indexed label map")
    ori_w, ori_h = seg.size
    th, tw = scaled_size(ori_h, ori_w, shorter=shorter)
    if th % stride or tw % stride:
        raise RuntimeError(f"scaled frame {(th, tw)} is not divisible by stride {stride}")
    small = np.array(seg.resize((tw // stride, th // stride), _NEAREST))
    labels = torch.tensor(small, dtype=torch.float32).unsqueeze(0)
    return to_one_hot(labels), original


def voc_palette() -> list[int]:
    """PASCAL / DAVIS 256-color palette (index 0 is black, index 1 is dark red)."""
    palette = [0] * (256 * 3)
    for index in range(256):
        red = green = blue = 0
        color = index
        for shift in range(8):
            red |= ((color >> 0) & 1) << (7 - shift)
            green |= ((color >> 1) & 1) << (7 - shift)
            blue |= ((color >> 2) & 1) << (7 - shift)
            color >>= 3
        palette[3 * index : 3 * index + 3] = (red, green, blue)
    return palette


def imwrite_indexed(path: Path, labels: np.ndarray, palette: list[int]) -> None:
    if labels.ndim != 2:
        raise ValueError("indexed PNG needs a 2D label map")
    if int(labels.max()) > 255:
        raise ValueError("label ids must fit in a PNG palette (0..255)")
    image = Image.fromarray(labels.astype(np.uint8), mode="P")
    image.putpalette(palette)
    image.save(path, format="PNG")


def norm_mask(mask: torch.Tensor) -> torch.Tensor:
    """Per-class min-max, as in the DINO script. Flat channels are left as-is."""
    for channel in range(mask.shape[0]):
        plane = mask[channel]
        high = plane.max()
        low = plane.min()
        if high > 0 and high > low:
            mask[channel] = (plane - low) / (high - low)
    return mask


def annotation_path(frame: Path) -> Path:
    parts = ["Annotations" if part == "JPEGImages" else part for part in frame.parts]
    return Path(*parts).with_suffix(".png")


def seg2bmap(seg: np.ndarray) -> np.ndarray:
    """1-pixel boundaries used by the DAVIS contour score."""
    seg = np.asarray(seg, dtype=bool)
    east = np.zeros_like(seg)
    south = np.zeros_like(seg)
    south_east = np.zeros_like(seg)
    east[:, :-1] = seg[:, 1:]
    south[:-1, :] = seg[1:, :]
    south_east[:-1, :-1] = seg[1:, 1:]
    boundary = seg ^ east | seg ^ south | seg ^ south_east
    boundary[-1, :] = seg[-1, :] ^ east[-1, :]
    boundary[:, -1] = seg[:, -1] ^ south[:, -1]
    boundary[-1, -1] = False
    return boundary


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def dilate_disk(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    structure = _disk(radius)
    if ndimage is not None:
        return ndimage.binary_dilation(mask, structure=structure)
    # ponytail: shift-OR disk, used only when scipy is absent. Same footprint.
    height, width = mask.shape
    out = np.zeros((height, width), dtype=bool)
    for dy, dx in zip(*np.nonzero(structure)):
        dy -= radius
        dx -= radius
        y_src = slice(max(0, -dy), min(height, height - dy))
        x_src = slice(max(0, -dx), min(width, width - dx))
        y_dst = slice(max(0, dy), min(height, height + dy))
        x_dst = slice(max(0, dx), min(width, width + dx))
        out[y_dst, x_dst] |= mask[y_src, x_src]
    return out


def db_eval_iou(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Region similarity J. Both arrays are binary."""
    prediction = np.asarray(prediction, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    union = np.logical_or(prediction, ground_truth).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, ground_truth).sum() / union)


def db_eval_boundary(prediction: np.ndarray, ground_truth: np.ndarray, bound_th: float = 0.008) -> float:
    """Contour accuracy F. ``bound_th`` is the DAVIS tolerance as a fraction of the diagonal."""
    prediction = np.asarray(prediction, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    bound_pix = bound_th if bound_th >= 1 else int(np.ceil(bound_th * np.linalg.norm(prediction.shape)))
    pred_b = seg2bmap(prediction)
    gt_b = seg2bmap(ground_truth)
    pred_match = pred_b & dilate_disk(gt_b, bound_pix)
    gt_match = gt_b & dilate_disk(pred_b, bound_pix)
    n_pred = int(pred_b.sum())
    n_gt = int(gt_b.sum())
    if n_pred == 0 and n_gt == 0:
        precision, recall = 1.0, 1.0
    elif n_pred == 0:
        precision, recall = 1.0, 0.0
    elif n_gt == 0:
        precision, recall = 0.0, 1.0
    else:
        precision = float(pred_match.sum() / n_pred)
        recall = float(gt_match.sum() / n_gt)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def list_videos(data_path: Path, split: str) -> list[str]:
    path = data_path / split
    if not path.is_file():
        raise FileNotFoundError(f"missing DAVIS split list {path}")
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names:
        raise RuntimeError(f"no videos in {path}")
    return names


def segment_video(
    model: nn.Module,
    frames: list[Path],
    output_dir: Path,
    *,
    stride: int,
    n_last_frames: int,
    radius: int | None,
    topk: int,
    device: torch.device,
    shorter: int,
    palette: list[int],
) -> None:
    """Propagate the first annotation through ``frames`` and write indexed PNGs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    first_seg, first_labels = read_seg(annotation_path(frames[0]), stride, shorter=shorter)
    imwrite_indexed(output_dir / f"{frames[0].stem}.png", first_labels, palette)

    frame0, _, _ = read_frame(frames[0], shorter=shorter)
    first_tokens, height, width = extract_feature(model, frame0, device, return_hw=True)
    if first_seg.shape[-2:] != (height, width):
        raise RuntimeError(
            f"first-frame labels {tuple(first_seg.shape[-2:])} != feature map {(height, width)}"
        )
    if not torch.isfinite(first_tokens).all():
        raise RuntimeError("non-finite features on the first frame")
    first_seg = first_seg.to(device)
    context: list[tuple[torch.Tensor, torch.Tensor]] = []
    neighborhood = None

    for frame_path in tqdm(frames[1:], leave=False):
        frame, ori_h, ori_w = read_frame(frame_path, shorter=shorter)
        sources = [first_tokens.transpose(0, 1).contiguous()] + [pair[0] for pair in context]
        segs = [first_seg] + [pair[1] for pair in context]
        soft, tokens, neighborhood = label_propagation(
            model,
            frame,
            sources,
            segs,
            topk=topk,
            radius=radius,
            neighborhood=neighborhood,
            device=device,
        )
        if n_last_frames > 0:
            if len(context) == n_last_frames:
                context.pop(0)
            context.append((tokens, soft))
        try:
            up = F.interpolate(
                soft,
                scale_factor=stride,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        except TypeError:
            up = F.interpolate(soft, scale_factor=stride, mode="bilinear", align_corners=False)
        up = up[0]
        up = norm_mask(up)
        labels = up.argmax(dim=0).to(dtype=torch.uint8).cpu().numpy()
        labels = np.array(Image.fromarray(labels).resize((ori_w, ori_h), _NEAREST))
        imwrite_indexed(output_dir / f"{frame_path.stem}.png", labels, palette)


def score_video(pred_dir: Path, frames: list[Path]) -> dict[str, float] | None:
    """Mean per-object J and F. ``None`` when any ground-truth frame is missing."""
    gt_paths = [annotation_path(frame) for frame in frames]
    if not all(path.is_file() for path in gt_paths):
        return None
    ground_truth = [np.array(Image.open(path)) for path in gt_paths]
    objects = [int(value) for value in np.unique(ground_truth[0]) if int(value) != 0]
    if not objects:
        return None
    per_j: list[float] = []
    per_f: list[float] = []
    for obj in objects:
        js: list[float] = []
        fs: list[float] = []
        for frame, gt in zip(frames, ground_truth):
            pred = np.array(Image.open(pred_dir / f"{frame.stem}.png"))
            if pred.shape != gt.shape:
                raise RuntimeError(f"{frame.stem}: prediction {pred.shape} != annotation {gt.shape}")
            js.append(db_eval_iou(pred == obj, gt == obj))
            fs.append(db_eval_boundary(pred == obj, gt == obj))
        per_j.append(float(np.mean(js)))
        per_f.append(float(np.mean(fs)))
    return {
        "J": float(np.mean(per_j)),
        "F": float(np.mean(per_f)),
        "J&F": float(0.5 * (np.mean(per_j) + np.mean(per_f))),
        "objects": float(len(objects)),
        "per_object_J": per_j,
        "per_object_F": per_f,
    }


def evaluate(args: argparse.Namespace, model: nn.Module | None = None) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if model is None:
        model = load_resnet50(args.pretrained_weights, args.stride)
    if int(getattr(model, "feature_stride", args.stride)) != args.stride:
        raise RuntimeError(f"model stride {model.feature_stride} != --stride {args.stride}")
    model.to(device)
    model.eval()
    radius = args.size_mask_neighborhood if args.size_mask_neighborhood > 0 else None
    names = list_videos(args.data_path, args.split)
    if args.video:
        names = [name for name in names if name == args.video]
        if not names:
            raise RuntimeError(f"{args.video} is not in {args.split}")
    palette = voc_palette()
    shorter = int(getattr(args, "shorter", SHORTER))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    object_j: list[float] = []
    object_f: list[float] = []
    videos: list[dict] = []

    print(
        f"resnet50 stride={args.stride} device={device} videos={len(names)} "
        f"n_last_frames={args.n_last_frames} radius={args.size_mask_neighborhood} topk={args.topk}"
    )
    for index, name in enumerate(names):
        video_dir = args.data_path / "JPEGImages" / "480p" / name
        frames = sorted(video_dir.glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(f"no jpeg frames in {video_dir}")
        print(f"[{index + 1}/{len(names)}] {name} ({len(frames)} frames)")
        pred_dir = output_dir / name
        with torch.no_grad():
            segment_video(
                model,
                frames,
                pred_dir,
                stride=args.stride,
                n_last_frames=args.n_last_frames,
                radius=radius,
                topk=args.topk,
                device=device,
                shorter=shorter,
                palette=palette,
            )
        scored = score_video(pred_dir, frames)
        if scored is None:
            print("  masks only (annotations incomplete)")
            videos.append({"video": name, "frames": len(frames), "scored": False})
            continue
        # Dataset J&F averages objects, not videos: a sequence with more objects counts more.
        object_j.extend(scored["per_object_J"])
        object_f.extend(scored["per_object_F"])
        print(
            f"  J {100 * scored['J']:.1f}  F {100 * scored['F']:.1f}  "
            f"J&F {100 * scored['J&F']:.1f}  objects {int(scored['objects'])}"
        )
        videos.append(
            {
                "video": name,
                "frames": len(frames),
                "scored": True,
                "objects": int(scored["objects"]),
                "J": scored["J"],
                "F": scored["F"],
                "J&F": scored["J&F"],
            }
        )

    summary = {
        "arch": "resnet50",
        "stride": args.stride,
        "checkpoint": None if args.pretrained_weights is None else str(args.pretrained_weights),
        "protocol": {
            "shorter": shorter,
            "n_last_frames": args.n_last_frames,
            "size_mask_neighborhood": args.size_mask_neighborhood,
            "topk": args.topk,
            "temperature": TEMPERATURE,
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        },
        "videos": videos,
    }
    if object_j:
        j_mean = float(np.mean(object_j))
        f_mean = float(np.mean(object_f))
        summary["J"] = j_mean
        summary["F"] = f_mean
        summary["J&F"] = 0.5 * (j_mean + f_mean)
        summary["num_objects"] = len(object_j)
        print(
            f"DAVIS mean over {len(object_j)} objects: "
            f"J {100 * j_mean:.1f}  F {100 * f_mean:.1f}  J&F {100 * summary['J&F']:.1f}"
        )
    out_json = output_dir / "davis2017.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_json}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained-weights", type=Path, help="SimCLR/SimLAP/MoCo checkpoint or ResNet-50 backbone")
    parser.add_argument("--data-path", type=Path, help="DAVIS root (JPEGImages, Annotations, ImageSets)")
    parser.add_argument("--output-dir", type=Path, default=Path("davis-resnet50"))
    parser.add_argument("--stride", type=int, default=32, choices=STRIDES, help="layer4 stride in pixels (DINO patch_size)")
    parser.add_argument("--split", default="ImageSets/2017/val.txt")
    parser.add_argument("--video", default="", help="run a single sequence name from the split list")
    parser.add_argument("--n-last-frames", type=int, default=7, help="preceding frames kept besides the first")
    parser.add_argument(
        "--size-mask-neighborhood",
        type=int,
        default=12,
        help="spatial radius on the feature grid; 0 disables the mask (DINO default 12)",
    )
    parser.add_argument("--topk", type=int, default=5, help="source locations pooled per query")
    parser.add_argument("--device", default=None)
    parser.add_argument("--self-check", action="store_true", help="run the CPU sanity check and exit")
    return parser.parse_args(argv)


def _brute_neighborhood(height: int, width: int, radius: int) -> torch.Tensor:
    mask = torch.zeros(height * width, height * width)
    for i in range(height):
        for j in range(width):
            for p in range(height):
                for q in range(width):
                    if abs(i - p) <= radius and abs(j - q) <= radius:
                        mask[i * width + j, p * width + q] = 1
    return mask


def _self_check() -> None:
    """Fail if feature stride, propagation, or DAVIS scores drift."""
    assert scaled_size(480, 854) == (480, 832)
    assert scaled_size(854, 480) == (832, 480)
    assert voc_palette()[3:6] == [128, 0, 0]
    mask = restrict_neighborhood(3, 4, 1, torch.device("cpu")).to(dtype=torch.float32)
    assert torch.equal(mask, _brute_neighborhood(3, 4, 1))

    blob = np.zeros((32, 32), dtype=bool)
    blob[5:20, 5:20] = True
    other = np.zeros_like(blob)
    other[5:20, 22:30] = True
    empty = np.zeros_like(blob)
    assert db_eval_iou(blob, blob) == 1.0
    assert db_eval_boundary(blob, blob) == 1.0
    assert db_eval_iou(blob, other) == 0.0
    assert db_eval_boundary(blob, other) == 0.0
    assert db_eval_iou(empty, empty) == 1.0
    assert db_eval_boundary(empty, empty) == 1.0

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        labels = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        png = Path(tmp) / "labels.png"
        imwrite_indexed(png, labels, voc_palette())
        assert np.array_equal(np.array(Image.open(png)), labels)

    device = torch.device("cpu")
    model = make_resnet50(32)
    model.fc = nn.Identity()
    model.eval()
    wrapped = {f"backbone.{key}": value for key, value in model.state_dict().items()}
    wrapped["projector.0.weight"] = torch.zeros(4, 4)
    state = extract_backbone(wrapped)
    assert "conv1.weight" in state and "projector.0.weight" not in state
    other = make_resnet50(16)
    other.fc = nn.Identity()
    other.load_state_dict(model.state_dict(), strict=True)
    sample = torch.rand(1, 3, 64, 128)
    assert forward_features(model, sample).shape[-2:] == (2, 4)
    assert forward_features(other, sample).shape[-2:] == (4, 8)
    stride8 = make_resnet50(8)
    stride8.eval()
    assert forward_features(stride8, sample).shape[-2:] == (8, 16)

    frame = torch.rand(3, 64, 128)
    tokens, height, width = extract_feature(model, frame, device, return_hw=True)
    assert (height, width) == (2, 4)
    seg = torch.zeros(1, 2, height, width)
    seg[0, 1, :, :2] = 1
    seg[0, 0, :, 2:] = 1
    out, stored, neighborhood = label_propagation(
        model,
        frame,
        [tokens.transpose(0, 1).contiguous()],
        [seg],
        topk=1000,
        radius=0,
        neighborhood=None,
        device=device,
    )
    assert stored.shape == (2048, height * width)
    assert neighborhood is not None and neighborhood.shape == (height * width, height * width)
    assert torch.allclose(out, seg, atol=1e-5)
    assert torch.allclose(out.sum(dim=1), torch.ones(1, height, width), atol=1e-5)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        video = root / "JPEGImages" / "480p" / "toy"
        ann = root / "Annotations" / "480p" / "toy"
        video.mkdir(parents=True)
        ann.mkdir(parents=True)
        (root / "ImageSets" / "2017").mkdir(parents=True)
        (root / "ImageSets" / "2017" / "val.txt").write_text("toy\n")
        rgb = np.zeros((64, 96, 3), dtype=np.uint8)
        rgb[:32, :, 0] = 220
        rgb[32:, :, 1] = 180
        lab = np.zeros((64, 96), dtype=np.uint8)
        lab[:, 40:] = 1
        for name, shift in (("00000", 0), ("00001", 3)):
            Image.fromarray(np.roll(rgb, shift, axis=1)).save(video / f"{name}.jpg")
            Image.fromarray(np.roll(lab, shift, axis=1)).save(ann / f"{name}.png")
        args = parse_args(
            [
                "--data-path",
                str(root),
                "--output-dir",
                str(root / "out"),
                "--stride",
                "32",
                "--device",
                "cpu",
                "--n-last-frames",
                "1",
                "--topk",
                "5",
            ]
        )
        args.shorter = 64
        summary = evaluate(args, model=model)
        first = np.array(Image.open(root / "out" / "toy" / "00000.png"))
        assert np.array_equal(first, lab)
        assert (root / "out" / "toy" / "00001.png").is_file()
        assert 0.0 <= summary["J&F"] <= 1.0
    print("ok video segmentation")


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check()
        return
    if args.pretrained_weights is None or args.data_path is None:
        raise SystemExit("need --pretrained-weights and --data-path (or --self-check)")
    evaluate(args)


if __name__ == "__main__":
    main()
