"""Precompute 1000 ImageNet class embeddings for X-CLR (Sobal et al. 2024).

Paper §4.1 / A.7: caption = "a photo of a {class}", Sentence Transformer
once, then G_ij = cosine(e[y_i], e[y_j]) via the C×C gram matrix.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

CAPTION = "a photo of a {}"
DEFAULT_TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"
BUNDLED_IN1K = Path(__file__).resolve().parent / "imagenet1k_xclr_graph.pt"
STL10_NAMES = (
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
)


def imagenet1k_names() -> list[str]:
    from torchvision.models import ResNet50_Weights

    return list(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])


def class_names_for(data_set: str) -> list[str]:
    if data_set == "STL":
        return list(STL10_NAMES)
    return imagenet1k_names()


def class_captions(names: list[str]) -> list[str]:
    return [CAPTION.format(name.replace("_", " ")) for name in names]


def offdiag_mean(sim: torch.Tensor) -> float:
    n = sim.size(0)
    return float((sim.sum() - sim.diag().sum()) / max(n * (n - 1), 1))


def class_sim_from_embeds(embeds: torch.Tensor) -> torch.Tensor:
    z = F.normalize(embeds.float(), dim=1)
    return z @ z.T


def load_bundle(path: str | Path) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        return {"class_sim": obj.float()}
    return obj


def load_class_embeds(path: str | Path) -> torch.Tensor:
    obj = load_bundle(path)
    if "embeds" not in obj:
        raise ValueError(f"{path} has no 'embeds' (C×D). Re-run: python -m cl.xclr_graph")
    emb = obj["embeds"].float()
    if emb.ndim != 2:
        raise ValueError(f"embeds must be C×D, got {tuple(emb.shape)}")
    return emb


def load_class_sim(path: str | Path) -> torch.Tensor:
    obj = load_bundle(path)
    if "embeds" in obj:
        sim = class_sim_from_embeds(obj["embeds"])
    elif "class_sim" in obj:
        sim = obj["class_sim"].float()
    else:
        raise ValueError(f"{path} needs 'embeds' or 'class_sim'")
    if sim.ndim != 2 or sim.size(0) != sim.size(1):
        raise ValueError(f"class_sim must be square, got {tuple(sim.shape)}")
    return sim


def encode_class_embeds(names: list[str], model_name: str = DEFAULT_TEXT_MODEL) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name)
    emb = encoder.encode(class_captions(names), convert_to_tensor=True, normalize_embeddings=True)
    return emb.cpu().float()


def save_class_embeds(
    path: str | Path, embeds: torch.Tensor, names: list[str], model_name: str
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    captions = class_captions(names)
    torch.save(
        {
            "embeds": embeds.cpu().float(),
            "names": names,
            "captions": captions,
            "model": model_name,
        },
        path,
    )


def resolve_class_sim(
    num_classes: int,
    names: list[str] | None = None,
    graph_path: str | None = None,
    text_model: str = DEFAULT_TEXT_MODEL,
    cache_path: str | Path | None = None,
) -> torch.Tensor:
    if graph_path:
        sim = load_class_sim(graph_path)
    elif cache_path and Path(cache_path).is_file():
        sim = load_class_sim(cache_path)
    elif num_classes == 1000 and BUNDLED_IN1K.is_file() and graph_path is None:
        sim = load_class_sim(BUNDLED_IN1K)
    else:
        names = names or (imagenet1k_names() if num_classes == 1000 else None)
        if not names or len(names) != num_classes:
            raise ValueError(f"X-CLR needs {num_classes} class names or --xclr-graph")
        try:
            embeds = encode_class_embeds(names, text_model)
        except ImportError as exc:
            raise SystemExit(
                "X-CLR embeds missing. Install sentence-transformers and run "
                "`python -m cl.xclr_graph`, or pass --xclr-graph"
            ) from exc
        dest = cache_path or (BUNDLED_IN1K if num_classes == 1000 else None)
        if dest:
            save_class_embeds(dest, embeds, names, text_model)
        sim = class_sim_from_embeds(embeds)
    if sim.shape != (num_classes, num_classes):
        raise ValueError(f"class_sim {tuple(sim.shape)} != ({num_classes}, {num_classes})")
    return sim


def main() -> None:
    names = imagenet1k_names()
    print("encoding", len(names), "class captions with", DEFAULT_TEXT_MODEL)
    embeds = encode_class_embeds(names)
    save_class_embeds(BUNDLED_IN1K, embeds, names, DEFAULT_TEXT_MODEL)
    sim = class_sim_from_embeds(embeds)
    print(
        f"wrote {BUNDLED_IN1K} embeds={tuple(embeds.shape)} "
        f"offdiag={offdiag_mean(sim):.3f} (paper ImageNet ~0.35)"
    )


if __name__ == "__main__":
    main()
