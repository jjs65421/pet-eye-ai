"""Train one binary pet-eye disease screening model from the AI Hub label archives.

The labels archive already includes cropped eye photos.  This script reads each
JSON file beside its JPG instead of duplicating the 200GB data set.
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import tensorflow as tf


DEFAULT_DATA_ROOT = Path(r"D:\153.반려동물 안구질환 데이터\extracted")
SMOKE_DATA_ROOT = Path(r"D:\153.반려동물 안구질환 데이터\smoke-cataract")
IMAGE_SIZE = 224
DISEASE_ALIASES = {
    "cataract": "백내장",
    "epiphora": "유루증",
    "sclerosis": "핵경화",
    "blepharitis": "안검염",
    # The downloaded AI Hub labels use these names (not "각막염" / "각막격리증").
    "conjunctivitis": "결막염",
    "corneal_ulcer": "각막궤양",
}


def is_abnormal(value):
    return str(value).strip() not in {"", "무", "정상", "없음", "none", "None", "0"}


def samples_for(root, disease):
    paths, targets = [], []
    for json_path in root.rglob("*.json"):
        try:
            with json_path.open("r", encoding="utf-8-sig") as source:
                annotation = json.load(source)
            label = annotation["label"]
            if label.get("label_deleted") or label.get("label_disease_nm") != disease:
                continue
            image_path = json_path.with_name(label["label_filename"])
            if not image_path.is_file():
                continue
            paths.append(str(image_path))
            targets.append(int(is_abnormal(label.get("label_disease_lv_1"))))
        except (OSError, UnicodeDecodeError, ValueError, KeyError):
            continue
    return paths, targets


def samples_from_manifest(manifest_path, split, disease):
    paths, targets = [], []
    with manifest_path.open("r", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["split"] == split and record["disease"] == disease:
                paths.append(record["image_path"])
                targets.append(record["abnormal"])
    return paths, targets


def dataset(paths, targets, training):
    data = tf.data.Dataset.from_tensor_slices((paths, targets))

    def load(path, target):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        return image, tf.cast(target, tf.float32)

    data = data.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        data = data.shuffle(min(len(paths), 10_000), seed=42)
        data = data.map(
            lambda image, target: (tf.image.random_flip_left_right(image), target),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return data.batch(16).prefetch(tf.data.AUTOTUNE)


def make_model():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), pooling="avg"
    )
    base.trainable = False
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    x = tf.keras.layers.RandomRotation(0.04)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="abnormal_probability")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", required=True, help="예: 백내장")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="빠른 시연용 샘플 제한. 0이면 전체")
    parser.add_argument("--smoke", action="store_true", help="빠른 검증용 표본 데이터 사용")
    args = parser.parse_args()
    args.disease = DISEASE_ALIASES.get(args.disease.lower(), args.disease)
    if args.smoke:
        args.data_root = SMOKE_DATA_ROOT

    manifest_path = args.data_root / "labels_manifest.jsonl"
    if manifest_path.is_file() and not args.smoke:
        train_paths, train_targets = samples_from_manifest(manifest_path, "training", args.disease)
        valid_paths, valid_targets = samples_from_manifest(manifest_path, "validation", args.disease)
    else:
        train_label_root = args.data_root / "training" / "labels"
        if not train_label_root.is_dir():
            train_label_root = args.data_root / "train" / "labels"
        train_paths, train_targets = samples_for(train_label_root, args.disease)
        valid_paths, valid_targets = samples_for(args.data_root / "validation" / "labels", args.disease)
    if args.limit:
        train_paths, train_targets = train_paths[:args.limit], train_targets[:args.limit]
        valid_limit = max(100, args.limit // 5)
        valid_paths, valid_targets = valid_paths[:valid_limit], valid_targets[:valid_limit]
    if not train_paths or not valid_paths or len(set(train_targets)) < 2:
        raise SystemExit(f"'{args.disease}'의 정상/이상 학습 표본이 충분하지 않습니다: {Counter(train_targets)}")

    print("train", len(train_paths), Counter(train_targets))
    print("validation", len(valid_paths), Counter(valid_targets))
    slug = args.disease.replace(" ", "_")
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{slug}.keras"
    weights = {0: len(train_targets) / (2 * train_targets.count(0)), 1: len(train_targets) / (2 * train_targets.count(1))}
    model = make_model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=2, restore_best_weights=True),
    ]
    model.fit(
        dataset(train_paths, train_targets, True),
        validation_data=dataset(valid_paths, valid_targets, False),
        epochs=args.epochs,
        class_weight=weights,
        callbacks=callbacks,
    )
    with (model_dir / f"{slug}.json").open("w", encoding="utf-8") as metadata:
        json.dump({"disease": args.disease, "image_size": IMAGE_SIZE, "positive": "이상 가능성"}, metadata, ensure_ascii=False)
    print(f"Saved {model_path}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    main()
