"""Evaluate the approved saved screening models on the full validation split."""
import json
from pathlib import Path

import tensorflow as tf

from train import DEFAULT_DATA_ROOT, dataset, samples_from_manifest


MODEL_DIR = Path(__file__).parent / "models"
DISEASES = ["백내장", "유루증", "핵경화", "안검염", "결막염"]


def main():
    manifest = DEFAULT_DATA_ROOT / "labels_manifest.jsonl"
    report = {}
    for disease in DISEASES:
        paths, targets = samples_from_manifest(manifest, "validation", disease)
        model = tf.keras.models.load_model(MODEL_DIR / f"{disease}.keras")
        probabilities = model.predict(dataset(paths, targets, False), verbose=0).reshape(-1)
        labels = tf.convert_to_tensor(targets, dtype=tf.float32)
        predictions = tf.convert_to_tensor(probabilities, dtype=tf.float32)
        accuracy = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        auc = tf.keras.metrics.AUC(curve="ROC")
        accuracy.update_state(labels, predictions)
        auc.update_state(labels, predictions)
        report[disease] = {
            "validation_samples": len(targets),
            "accuracy_at_0_5": round(float(accuracy.result()), 4),
            "roc_auc": round(float(auc.result()), 4),
        }
        print(f"{disease}: {report[disease]}", flush=True)
    output = MODEL_DIR / "validation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
