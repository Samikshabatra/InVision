"""Steps 2 and 3 - CLIP image embeddings and RapidOCR signage text."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

_clip_cache: dict = {}
_ocr_cache: dict = {}


def _load_clip():
    if "model" not in _clip_cache:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(config.CLIP_MODEL)
        model.eval()
        _clip_cache["model"] = model
        _clip_cache["processor"] = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        _clip_cache["torch"] = torch
    return _clip_cache["model"], _clip_cache["processor"], _clip_cache["torch"]


def _load_ocr():
    if "engine" not in _ocr_cache:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_cache["engine"] = RapidOCR()
    return _ocr_cache["engine"]


def embed_images(paths: list[Path], batch_size: int = 16) -> tuple[np.ndarray, list[int]]:
    """Encode images to L2-normalised CLIP vectors.

    Returns the embedding matrix and the indices of images that failed to load,
    so callers can keep row alignment with a filtered path list.
    """
    model, processor, torch = _load_clip()

    vectors: list[np.ndarray] = []
    failed: list[int] = []
    pending: list[tuple[int, Image.Image]] = []

    embed_dim = model.visual_projection.out_features
    proj_in_dim = model.visual_projection.in_features

    def _to_embedding(value):
        """Project to CLIP's shared space only if not already projected."""
        if value.ndim == 3:  # sequence output, take the CLS token
            value = value[:, 0]
        if value.shape[-1] == embed_dim:
            return value
        if value.shape[-1] == proj_in_dim:
            return model.visual_projection(value)
        raise RuntimeError(f"unexpected CLIP feature width {value.shape[-1]}")

    def _project(inputs):
        """Return image embeddings across transformers versions.

        get_image_features yields a bare tensor on v4 but a model-output
        wrapper on v5, whose pooled tensor is already projected.
        """
        feats = model.get_image_features(**inputs)
        if hasattr(feats, "norm"):
            return _to_embedding(feats)
        for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
            value = getattr(feats, attr, None)
            if value is not None and hasattr(value, "norm"):
                return _to_embedding(value)
        vision = model.vision_model(**inputs)
        return _to_embedding(vision.pooler_output)

    def flush():
        if not pending:
            return
        images = [img for _, img in pending]
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            feats = _project(inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vectors.append(feats.cpu().numpy().astype(np.float32))
        pending.clear()

    for idx, path in enumerate(paths):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            failed.append(idx)
            continue
        pending.append((idx, img))
        if len(pending) >= batch_size:
            flush()
    flush()

    if not vectors:
        return np.zeros((0, 512), dtype=np.float32), failed
    return np.vstack(vectors), failed


_TEXT_NOISE = re.compile(r"[^a-z0-9 ]+")


def normalise_text(text: str) -> str:
    return _TEXT_NOISE.sub(" ", text.lower()).strip()


def ocr_image(path: Path, min_score: float = 0.5) -> dict:
    """Extract signage text from one image.

    Returns the concatenated normalised text plus the individual detections,
    which are kept so low-confidence reads can be re-weighted later.
    """
    engine = _load_ocr()
    try:
        result, _ = engine(str(path))
    except Exception as exc:
        return {"text": "", "tokens": [], "error": str(exc)}

    if not result:
        return {"text": "", "tokens": []}

    tokens = []
    for detection in result:
        # RapidOCR returns [box, text, score] per detection.
        if len(detection) < 3:
            continue
        box, text, score = detection[0], detection[1], detection[2]
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        if score < min_score or not text.strip():
            continue
        # Cap height is the strongest brand cue: shopfront names are set far
        # larger than promotional copy, so it is kept for later ranking.
        try:
            ys = [float(p[1]) for p in box]
            height = max(ys) - min(ys)
        except (TypeError, ValueError, IndexError):
            height = 0.0
        tokens.append(
            {"text": text.strip(), "score": round(score, 4), "height": round(height, 1)}
        )

    joined = normalise_text(" ".join(t["text"] for t in tokens))
    return {"text": joined, "tokens": tokens}
