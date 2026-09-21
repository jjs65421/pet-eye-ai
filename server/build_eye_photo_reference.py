"""Create a reference for deciding whether an upload resembles a pet-eye photo.

This is an input-quality gate, not a diagnosis model. It learns the visual
neighbourhood of labelled pet-eye images already used by this project.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

from train import DEFAULT_DATA_ROOT, dataset


MODEL_PATH = Path(__file__).parent / "models" / "백내장.keras"
OUTPUT_PATH = Path(__file__).parent / "models" / "eye_photo_reference.npz"
SAMPLE_COUNT = 1200


def main():
    manifest = DEFAULT_DATA_ROOT / "labels_manifest.jsonl"
    paths = []
    with manifest.open("r", encoding="utf-8") as source:
        for line in source:
            record = __import__("json").loads(line)
            if record["split"] == "training":
                paths.append(record["image_path"])
    paths = paths[::max(1, len(paths) // SAMPLE_COUNT)][:SAMPLE_COUNT]
    model = tf.keras.models.load_model(MODEL_PATH)
    # The nested EfficientNet input is disconnected after model deserialization;
    # the outer dropout output carries the same 1280-D feature vector at inference.
    extractor = tf.keras.Model(model.input, model.get_layer("dropout").output)
    features = extractor.predict(dataset(paths, [0] * len(paths), False), verbose=1)
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    centroid = features.mean(axis=0)
    centroid = centroid / np.maximum(np.linalg.norm(centroid), 1e-8)
    similarities = features @ centroid
    # A low percentile allows legitimate variation while rejecting unrelated photos.
    threshold = float(np.percentile(similarities, 2.0))
    np.savez(OUTPUT_PATH, centroid=centroid.astype("float32"), threshold=threshold)
    print({"samples": len(paths), "threshold": round(threshold, 4)}, flush=True)


if __name__ == "__main__":
    main()
