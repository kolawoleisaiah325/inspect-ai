"""Small, frozen CNN feature extractor + nearest normal-patch scoring.

This is a simplified patch-memory approach, not an exact PatchCore reproduction.
No torch dependency is needed for serving the exported ONNX feature extractor.
"""
import base64
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

SIZE = 160
MAX_BYTES = 2 * 1024 * 1024
MAX_PIXELS = 16_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

def decode_image(content):
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Choose a JPEG, PNG or WebP image smaller than 2 MB.")
    try:
        image = Image.open(io.BytesIO(content))
        if image.format not in {"JPEG", "PNG", "WEBP"} or image.width * image.height > MAX_PIXELS:
            raise ValueError("Use a JPEG, PNG or WebP image up to 16 megapixels.")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Choose a still image, rather than an animation.")
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("The image could not be decoded. Try a different JPEG, PNG or WebP.") from error
    if min(image.size) < 32:
        raise ValueError("The image is too small. Use at least 32 pixels on each side.")
    return image

def preprocess(image):
    pixels = np.asarray(image.resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    normalized = (pixels - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return normalized.transpose(2, 0, 1)[None].astype(np.float32)

def patch_distances(features, bank, scale):
    flat = features[0].transpose(1, 2, 0).reshape(-1, features.shape[1]) / scale
    # Batch bounds temporary memory, and float rounding can only make tiny negatives.
    values = []
    bank_norm = np.sum(bank * bank, axis=1)
    for start in range(0, len(flat), 64):
        block = flat[start:start + 64]
        squared = np.sum(block * block, axis=1)[:, None] + bank_norm[None] - 2 * block @ bank.T
        values.append(np.sqrt(np.maximum(squared.min(axis=1), 0)))
    return np.concatenate(values).reshape(features.shape[2:])

def heatmap_png(scores, ceiling):
    strength = np.clip(scores / ceiling, 0, 1)
    # A fixed model-wide scale; we never exaggerate a normal image with per-image min/max.
    red = np.clip(2 * strength, 0, 1)
    green = np.clip(2 * (1 - np.abs(strength - .5) * 2), 0, 1)
    blue = np.clip(1 - strength * 2, 0, 1)
    alpha = np.clip((strength - .20) / .65, 0, 1) * .85
    rgba = np.stack([red, green, blue, alpha], axis=-1)
    image = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").resize((320, 320), Image.Resampling.BILINEAR)
    stream = io.BytesIO(); image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")

class Inspector:
    def __init__(self, directory=None):
        root = Path(directory or Path(__file__).parent / "artifacts")
        self.metadata = json.loads((root / "metadata.json").read_text())
        arrays = np.load(root / "memory.npz", allow_pickle=False)
        self.bank = arrays["bank"].astype(np.float32)
        self.scale = arrays["scale"].astype(np.float32)
        options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(root / "features.onnx"), options, providers=["CPUExecutionProvider"])
        self.lock = threading.Lock()

    def score_image(self, image):
        features = self.session.run(None, {"image": preprocess(image)})[0]
        field = patch_distances(features, self.bank, self.scale)
        # One damaged region should be visible, while averaging 3 highest patches reduces single-patch noise.
        score = float(np.sort(field.ravel())[-3:].mean())
        return score, field

    def inspect(self, content):
        image = decode_image(content)
        started = time.perf_counter()
        with self.lock: score, field = self.score_image(image)
        threshold = self.metadata["threshold"]
        return {"model_version": self.metadata["model_version"], "score": round(score, 5),
                "threshold": threshold, "relative_score": round(score / threshold, 3),
                "decision": "review" if score >= threshold else "no_anomaly_flagged",
                "inference_ms": round((time.perf_counter() - started) * 1000, 1),
                "heatmap": "data:image/png;base64," + heatmap_png(field, self.metadata["heatmap_ceiling"]),
                "heatmap_scale": "Fixed calibration scale", "image_size": list(image.size),
                "notice": "Bottle benchmark prototype. An anomaly score is not a defect probability. Unfamiliar products, backgrounds and camera angles may produce unreliable results."}
