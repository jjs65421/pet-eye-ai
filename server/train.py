"""Confirmed-label transfer learning with shared, cached EfficientNet features."""
import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from image_pipeline import IMAGE_SIZE, PREPROCESSING, pixels_from_path
from model_loading import load_screening_model
from screening_data import (DEFAULT_DATA_ROOT, MANIFEST_PATH, DISEASES,
                            DISEASE_ALIASES, SERVER_ROOT, annotation_record, read_records, partition)
from screening_policy import select_thresholds, validate_cutoffs

SMOKE_DATA_ROOT = SERVER_ROOT / 'training_data' / 'smoke-cataract'


def samples_for(root, disease):
    rows = []
    for path in Path(root).rglob("*.json"):
        row = annotation_record(path, "training")
        if row and row["disease"] == disease:
            rows.append(row)
    return [r["image_path"] for r in rows], [r["abnormal"] for r in rows]


def samples_from_manifest(path, split, disease):
    rows = [r for r in read_records(path) if r["split"] == split and r["disease"] == disease]
    return [r["image_path"] for r in rows], [r["abnormal"] for r in rows]


def dataset(paths, targets, training=False, batch_size=32):
    data = tf.data.Dataset.from_tensor_slices((paths, targets))
    def load(path, target):
        image = tf.numpy_function(pixels_from_path, [path], tf.float32)
        image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
        return image, tf.cast(target, tf.float32)
    data = data.map(load, num_parallel_calls=4)
    if training:
        data = data.shuffle(min(len(paths), 10000), seed=42)
    return data.batch(batch_size).prefetch(2)


def backbone():
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), pooling="avg")
    base.trainable = False
    return base


def cached_features(rows, base):
    # Different diseases can refer to the same original image. Decode once per path.
    paths = sorted({r["image_path"] for r in rows})
    signature = hashlib.sha256((PREPROCESSING + "\n" + "\n".join(paths)).encode()).hexdigest()[:16]
    folder = SERVER_ROOT / "training_data"
    folder.mkdir(exist_ok=True)
    cache = folder / f"efficientnet-{signature}.npy"
    marker = cache.with_suffix(".complete")
    if not marker.is_file():
        features = np.lib.format.open_memmap(str(cache), mode="w+", dtype="float32", shape=(len(paths), 1280))
        data = dataset(paths, [0] * len(paths), batch_size=32)
        offset = 0
        for images, _ in data:
            values = base(images, training=False).numpy()
            features[offset:offset + len(values)] = values
            offset += len(values)
            if offset % 1024 == 0 or offset == len(paths):
                print(f"features={offset}/{len(paths)}", flush=True)
        features.flush()
        marker.write_text(signature, encoding="utf-8")
        del features
    return np.load(str(cache), mmap_mode="r"), {path: i for i, path in enumerate(paths)}


def balanced_training(rows):
    """Equal class weight and device weight within each class.

    Single-class devices cannot pass per-device deployment validation.
    """
    groups = Counter((r["device"], r["abnormal"]) for r in rows)
    if len({r['abnormal'] for r in rows}) < 2:
        raise ValueError("Both confirmed labels are required for training.")
    devices = sorted({d for d, _ in groups})
    device_counts = {label: len({d for d, target in groups if target == label}) for label in (0, 1)}
    weights = np.array([len(rows) / (2 * device_counts[r['abnormal']] * groups[r['device'], r['abnormal']])
                        for r in rows], dtype="float32")
    return rows, weights, devices


def train_one(disease, splits, features, index, base, output, epochs):
    rows, weights, devices = balanced_training([r for r in splits["train"] if r["disease"] == disease])
    calibration = [r for r in splits["calibration"] if r["disease"] == disease and r["device"] in devices]
    testing = [r for r in splits["test"] if r["disease"] == disease and r["device"] in devices]
    if any(len({r["abnormal"] for r in part}) < 2 for part in (calibration, testing)):
        raise ValueError(f"{disease}: calibration/test lacks both labels.")
    def arrays(part):
        return (np.asarray(features[[index[r["image_path"]] for r in part]]),
                np.array([r["abnormal"] for r in part], dtype="float32"))
    x_train, y_train = arrays(rows)
    x_cal, y_cal = arrays(calibration)
    x_test, y_test = arrays(testing)
    head = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1280,)),
        tf.keras.layers.Dropout(.20),
        tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(.001)),
    ])
    head.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="binary_crossentropy",
                 metrics=[tf.keras.metrics.AUC(name="auc")])
    print(f"training={disease}; train={len(rows)} calibration={len(calibration)} test={len(testing)} devices={devices}", flush=True)
    history = head.fit(x_train, y_train, sample_weight=weights, validation_data=(x_cal, y_cal),
        batch_size=128, epochs=epochs, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True)])
    calibration_scores = head.predict(x_cal, verbose=0).reshape(-1)
    test_scores = head.predict(x_test, verbose=0).reshape(-1)
    thresholds = select_thresholds(y_cal, calibration_scores)
    aggregate_evaluation = validate_cutoffs(y_test, test_scores, thresholds)
    evaluation = dict(aggregate_evaluation)
    strata = {}
    for device in sorted({r["device"] for r in splits["test"] if r["disease"] == disease}):
        if device not in devices:
            strata[device] = {"approved": False, "reason": "not_supported_by_training"}
            continue
        mask = np.array([r["device"] == device for r in testing])
        cal_mask = np.array([r['device'] == device for r in calibration])
        device_cutoffs = select_thresholds(y_cal[cal_mask], calibration_scores[cal_mask])
        strata[device] = validate_cutoffs(y_test[mask], test_scores[mask], device_cutoffs)
        strata[device]['thresholds'] = device_cutoffs
    smartphone = strata.get("스마트폰", {"approved": False, "reason": "no_smartphone_test_samples"})
    evaluation['approved'] = any(result['approved'] for result in strata.values())
    evaluation['reason'] = 'device_policy_passed' if evaluation['approved'] else 'no_device_passed_independent_test'
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    model = tf.keras.Model(inputs, head(base(inputs, training=False)))
    # TensorFlow 2.7 treats .keras as HDF5 here. .h5 states that format explicitly.
    model_path = output / f"{disease}.h5"
    model.save(str(model_path), include_optimizer=False)
    restored = load_screening_model(model_path)
    check_batch = np.stack([pixels_from_path(row['image_path']) for row in calibration[:3]])
    saved_scores = restored(check_batch, training=False).numpy().reshape(-1)
    export_error = float(np.max(np.abs(saved_scores - calibration_scores[:3])))
    if export_error >= .0001:
        raise RuntimeError(f'Exported model differs from calibration: {disease}, error={export_error}')
    info = {"schema": "screening-v2", "disease": disease, "label_field": "label_disease_lv_3",
            "preprocessing": PREPROCESSING, "image_size": IMAGE_SIZE, "positive_class": 1,
            "score_meaning": "uncalibrated binary screening score", "thresholds": thresholds,
            "evaluation": evaluation, "aggregate_evaluation": aggregate_evaluation, "by_device": strata,
            "smartphone_validated": bool(evaluation['approved'] and smartphone["approved"]), "training_devices": devices,
            "training_counts_by_device_and_class": dict(Counter(f"{r['device']}|{r['abnormal']}" for r in rows)),
            "train_samples": len(rows), "calibration_samples": len(calibration),
            "epochs_completed": len(history.history["loss"])}
    info['export_score_max_error'] = export_error
    model_path.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(output / f"{disease}-test-predictions.npz", labels=y_test, scores=test_scores,
             devices=np.array([r["device"] for r in testing]))
    print(f"evaluated={disease}; {evaluation}; smartphone_validated={info['smartphone_validated']}", flush=True)
    return info


def run(diseases=DISEASES, epochs=30, rebuild_manifest=False, data_root=DEFAULT_DATA_ROOT):
    from build_manifest import build
    tf.keras.utils.set_random_seed(42) if hasattr(tf.keras.utils, "set_random_seed") else tf.random.set_seed(42)
    np.random.seed(42)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if rebuild_manifest or not MANIFEST_PATH.is_file():
        build(data_root)
    records = read_records()
    splits, audit = partition(records)
    base = backbone()
    all_rows = [r for name in splits for r in splits[name]]
    features, index = cached_features(all_rows, base)
    version = datetime.now(timezone.utc).strftime("confirmed-%Y%m%d-%H%M%S")
    output = SERVER_ROOT / "models" / version
    output.mkdir(parents=True)
    report = {"version": version, "preprocessing": PREPROCESSING, "split_audit": audit,
              "limitations": ["Image-level separation only; animal identity not available.",
                              "Thresholds are development policy, not clinical validation.",
                              "Phone testing includes a reserved subset of original training; see split audit.",
                              "Most diseases have no confirmed smartphone positives."], "models": {}}
    for disease in diseases:
        try:
            report["models"][disease] = train_one(disease, splits, features, index, base, output, epochs)
        except ValueError as error:
            report["models"][disease] = {"error": str(error), "evaluation": {"approved": False}}
            print(f"model_skipped={disease}: {error}", flush=True)
    # Reference is built exclusively from training images, using the same frozen base.
    reference_rows = splits["train"][::max(1, len(splits["train"]) // 4000)][:4000]
    values = np.asarray(features[[index[r["image_path"]] for r in reference_rows]])
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    centroid = values.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
    np.savez(output / "eye_photo_reference.npz", centroid=centroid,
             threshold=float(np.percentile(values @ centroid, 2)))
    base.save(str(output / "feature_extractor.h5"), include_optimizer=False)
    (output / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Publish only a complete five-model run; a one-model experiment never replaces it.
    if set(diseases) == set(DISEASES) and all((output / f"{d}.h5").is_file() for d in DISEASES):
        pointer = SERVER_ROOT / "models" / "active_models.json"
        pending = pointer.with_suffix(".pending.json")
        pending.write_text(json.dumps({"version": version}, ensure_ascii=False), encoding="utf-8")
        os.replace(pending, pointer)
        print(f"published={version}; restart_server_to_load", flush=True)
    else:
        print(f"not_published={version}; incomplete_or_single_disease_run", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--rebuild-manifest", action="store_true")
    args = parser.parse_args()
    disease = DISEASE_ALIASES.get(args.disease.lower(), args.disease)
    if disease not in DISEASES:
        parser.error("Unsupported disease for this screening pipeline.")
    run([disease], args.epochs, args.rebuild_manifest, args.data_root)


if __name__ == "__main__":
    main()
