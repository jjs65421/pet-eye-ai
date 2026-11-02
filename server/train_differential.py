"""Learn five disease labels jointly, rather than ranking binary abnormality scores.

This classifier is conditional on the five annotated diseases. It has no healthy
class and is not a clinical diagnosis or a validated smartphone detector.
"""
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
from screening_data import DISEASES, SERVER_ROOT, read_records, partition
from train import backbone, cached_features

CLASSES = list(DISEASES)


def classification_metrics(labels, probabilities):
    predicted = probabilities.argmax(axis=1)
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(matrix, (labels, predicted), 1)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    precision = np.diag(matrix) / np.maximum(matrix.sum(axis=0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    # Equal weight for each true disease, so cataract's larger test set cannot
    # make a model that always predicts cataract look successful.
    balanced_prediction_share = (matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)).mean(axis=0)
    return {'samples': len(labels), 'classes': CLASSES, 'confusion_matrix': matrix.tolist(),
            'accuracy': float(np.mean(labels == predicted)), 'balanced_accuracy': float(recall.mean()),
            'macro_f1': float(f1.mean()), 'recall_by_disease': dict(zip(CLASSES, recall.tolist())),
            'prediction_counts': dict(zip(CLASSES, matrix.sum(axis=0).tolist())),
            'balanced_prediction_share': dict(zip(CLASSES, balanced_prediction_share.tolist()))}


def run(epochs=60):
    np.random.seed(42)
    tf.random.set_seed(42)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    splits, audit = partition(read_records())
    all_rows = [r for part in splits.values() for r in part]
    base = backbone()
    features, index = cached_features(all_rows, base)
    positives = [r for r in splits['train'] if r['disease'] in CLASSES and r['abnormal'] == 1]
    counts = Counter((r['device'], r['disease']) for r in positives)
    devices = sorted({device for device, _ in counts
                      if all(counts[device, disease] >= 30 for disease in CLASSES)})
    if not devices:
        raise RuntimeError('No device has enough examples of all five diseases.')

    def subset(name):
        return [r for r in splits[name] if r['disease'] in CLASSES and r['abnormal'] == 1 and r['device'] in devices]

    def arrays(rows):
        labels = np.array([CLASSES.index(r['disease']) for r in rows], dtype='int32')
        return np.asarray(features[[index[r['image_path']] for r in rows]]), labels

    def weights(rows):
        distribution = Counter((r['device'], r['disease']) for r in rows)
        if any(distribution[device, disease] == 0 for device in devices for disease in CLASSES):
            raise RuntimeError('A split is missing a disease/device stratum.')
        return np.array([len(rows) / (len(CLASSES) * len(devices) * distribution[r['device'], r['disease']])
                         for r in rows], dtype='float32')

    training, calibration, testing = (subset(name) for name in ('train', 'calibration', 'test'))
    x_train, y_train = arrays(training)
    x_cal, y_cal = arrays(calibration)
    x_test, y_test = arrays(testing)
    head = tf.keras.Sequential([
        tf.keras.layers.Input((1280,)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(.001)),
        tf.keras.layers.Dropout(.30),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax', name='conditional_disease_scores'),
    ])
    head.compile(optimizer=tf.keras.optimizers.Adam(.0005), loss='sparse_categorical_crossentropy',
                 weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='balanced_strata_accuracy')])
    print(f'classes={CLASSES}; devices={devices}; train={len(training)} calibration={len(calibration)} test={len(testing)}', flush=True)
    history = head.fit(x_train, y_train, sample_weight=weights(training),
        validation_data=(x_cal, y_cal, weights(calibration)), batch_size=128, epochs=epochs, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)])
    probabilities = head.predict(x_test, verbose=0)
    metrics = classification_metrics(y_test, probabilities)
    by_device = {}
    for device in devices:
        mask = np.array([r['device'] == device for r in testing])
        by_device[device] = classification_metrics(y_test[mask], probabilities[mask])
    version = datetime.now(timezone.utc).strftime('differential-%Y%m%d-%H%M%S')
    output = SERVER_ROOT / 'models' / version
    output.mkdir(parents=True)
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    model = tf.keras.Model(inputs, head(base(inputs, training=False)))
    model.save(str(output / 'classifier.h5'), include_optimizer=False)
    restored = load_screening_model(output / 'classifier.h5')
    positions = [next(i for i, target in enumerate(y_test) if target == disease) for disease in range(len(CLASSES))]
    batch = np.stack([pixels_from_path(testing[i]['image_path']) for i in positions])
    actual = restored(batch, training=False).numpy()
    export_error = float(np.max(np.abs(actual - probabilities[positions])))
    if export_error >= .0001:
        raise RuntimeError(f'Exported classifier differs from held-out scores: {export_error}')
    accepted = (metrics['balanced_accuracy'] >= .60 and metrics['macro_f1'] >= .60
                and min(metrics['recall_by_disease'].values()) >= .40
                and max(metrics['balanced_prediction_share'].values()) <= .65)
    report = {'schema': 'differential-v1', 'version': version, 'preprocessing': PREPROCESSING,
              'classes': CLASSES, 'training_devices': devices, 'train_samples': len(training),
              'calibration_samples': len(calibration), 'epochs_completed': len(history.history['loss']),
              'test': metrics, 'by_device': by_device, 'split_audit': audit,
              'export_score_max_error': export_error, 'accepted_for_reference_display': bool(accepted),
              'score_meaning': 'conditional_distribution_among_five_annotated_disease_classes',
              'limitations': ['No healthy or unknown class. Scores do not establish disease presence.',
                              'No common smartphone positive data for all five diseases; smartphone accuracy unverified.',
                              'Original-image split only; patient identity unavailable.']}
    (output / 'classifier.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez(output / 'test-predictions.npz', labels=y_test, scores=probabilities,
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
    parser.add_argument('--epochs', type=int, default=60)
    args = parser.parse_args()
    run(args.epochs)
