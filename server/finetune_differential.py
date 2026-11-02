"""Train five diseases jointly, retaining frozen batch normalization statistics."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from image_pipeline import IMAGE_SIZE, PREPROCESSING, pixels_from_path
from model_loading import load_screening_model
from screening_data import SERVER_ROOT, read_records, partition
from train_differential import CLASSES, classification_metrics


def image_dataset(rows, training=False, weighted=False):
    distribution = Counter((r['device'], r['disease']) for r in rows)
    devices = {r['device'] for r in rows}
    paths = [r['image_path'] for r in rows]
    labels = [CLASSES.index(r['disease']) for r in rows]
    weights = [len(rows) / (len(CLASSES) * len(devices) * distribution[r['device'], r['disease']]) for r in rows]
    data = tf.data.Dataset.from_tensor_slices((paths, labels, np.array(weights, dtype='float32')))
    if training:
        data = data.shuffle(min(len(rows), 10000), seed=42)
    def load(path, label, weight):
        image = tf.numpy_function(pixels_from_path, [path], tf.float32)
        image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
        if training:
            image = tf.image.random_flip_left_right(image)
        return (image, label, weight) if weighted else (image, label)
    return data.map(load, num_parallel_calls=4).batch(16).prefetch(2)


def run(warm_start, epochs=8, per_stratum=2000, device=None):
    tf.random.set_seed(42)
    rng = np.random.RandomState(42)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    warm = json.loads((warm_start / 'classifier.json').read_text(encoding='utf-8'))
    devices = [device] if device else warm['training_devices']
    if any(value not in warm['training_devices'] for value in devices):
        raise ValueError('Requested device is not represented in the warm-start data.')
    splits, audit = partition(read_records())
    def subset(name):
        return [r for r in splits[name] if r['disease'] in CLASSES and r['abnormal'] == 1 and r['device'] in devices]
    training = []
    full_training = subset('train')
    for device in devices:
        for disease in CLASSES:
            rows = [r for r in full_training if r['device'] == device and r['disease'] == disease]
            selection = rng.permutation(len(rows))[:per_stratum]
            training.extend(rows[i] for i in selection)
    calibration, testing = subset('calibration'), subset('test')
    model = load_screening_model(warm_start / 'classifier.h5')
    if warm['classes'] != CLASSES:
        base = model.layers[1]
        old_head = model.layers[2]
        inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
        features = base(inputs, training=False)
        hidden = old_head.layers[0](features)
        hidden = tf.keras.layers.Dropout(.30)(hidden)
        outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax', name='joint_disease_scores')(hidden)
        model = tf.keras.Model(inputs, outputs)
    base = model.layers[1]
    base.trainable = True
    for layer in base.layers[:-35]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss='sparse_categorical_crossentropy',
                  weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='balanced_strata_accuracy')])
    print(f'fine_tuning train={len(training)} calibration={len(calibration)} test={len(testing)} epochs={epochs}', flush=True)
    history = model.fit(image_dataset(training, True, True), validation_data=image_dataset(calibration, False, True),
        epochs=epochs, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
    probabilities = model.predict(image_dataset(testing), verbose=0)
    labels = np.array([CLASSES.index(r['disease']) for r in testing])
    metrics = classification_metrics(labels, probabilities)
    by_device = {}
    for device in devices:
        mask = np.array([r['device'] == device for r in testing])
        by_device[device] = classification_metrics(labels[mask], probabilities[mask])
    version = datetime.now(timezone.utc).strftime('differential-%Y%m%d-%H%M%S')
    output = SERVER_ROOT / 'models' / version
    output.mkdir(parents=True)
    model.save(str(output / 'classifier.h5'), include_optimizer=False)
    restored = load_screening_model(output / 'classifier.h5')
    positions = [next(i for i, target in enumerate(labels) if target == disease) for disease in range(len(CLASSES))]
    images = np.stack([pixels_from_path(testing[i]['image_path']) for i in positions])
    error = float(np.max(np.abs(restored(images, training=False).numpy() - probabilities[positions])))
    if error >= .0001:
        raise RuntimeError(f'Exported classifier scores differ: {error}')
    accepted = (metrics['balanced_accuracy'] >= .60 and metrics['macro_f1'] >= .60
                and min(metrics['recall_by_disease'].values()) >= .40
                and max(metrics['balanced_prediction_share'].values()) <= .65)
    report = {'schema': 'differential-v1', 'version': version, 'preprocessing': PREPROCESSING,
              'classes': CLASSES, 'training_devices': devices, 'train_samples': len(training),
              'warm_start': warm['version'], 'epochs_completed': len(history.history['loss']),
              'test': metrics, 'by_device': by_device, 'split_audit': audit,
              'export_score_max_error': error, 'accepted_for_reference_display': bool(accepted),
              'architecture': 'five_class_softmax_with_finetuned_EfficientNet_last_35_layers_BN_frozen',
              'score_meaning': 'conditional_distribution_among_five_annotated_disease_classes',
              'limitations': ['No healthy or unknown class: highest score is always selected.',
                              'Training and evaluation restricted to general-camera positive disease photographs.',
                              'Five-disease smartphone accuracy has not been evaluated.',
                              'Original-image split only; patient identity unavailable.',
                              'Evaluation data have informed earlier model development; no external validation.']}
    (output / 'classifier.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez(output / 'test-predictions.npz', labels=labels, scores=probabilities,
             devices=np.array([r['device'] for r in testing]))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if accepted:
        pointer = SERVER_ROOT / 'models' / 'active_differential.json'
        pending = pointer.with_suffix('.pending.json')
        pending.write_text(json.dumps({'version': version}), encoding='utf-8')
        os.replace(pending, pointer)
        print(f'published_differential={version}', flush=True)
    else:
        print('not_published: joint classifier did not meet the development policy', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--warm-start', required=True, type=Path)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--per-stratum', type=int, default=2000)
    parser.add_argument('--device', choices=['일반카메라', '검안경'])
    args = parser.parse_args()
    run(args.warm_start, args.epochs, args.per_stratum, args.device)
