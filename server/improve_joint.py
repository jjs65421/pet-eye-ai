"""Two-stage candidate; select on calibration only and retain paired evaluation."""
import json
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from screening_data import SERVER_ROOT, read_records, partition
from image_pipeline import IMAGE_SIZE, PREPROCESSING, pixels_from_path
from model_loading import load_screening_model
from finetune_differential import image_dataset
from train_differential import CLASSES, classification_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--dog-breed-balanced', action='store_true')
args = parser.parse_args()
tf.random.set_seed(71)
np.random.seed(71)
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
pointer = SERVER_ROOT / 'models' / 'active_differential.json'
previous = json.loads(pointer.read_text())['version']
folder = SERVER_ROOT / 'models' / previous
warm = json.loads((folder / 'classifier.json').read_text(encoding='utf-8'))
splits, audit = partition(read_records())
rows = {k: [r for r in v if r['abnormal'] == 1 and r['device'] == '일반카메라' and r['disease'] in CLASSES] for k, v in splits.items()}
if args.dog_breed_balanced:
    # Original metadata label_path confirms D filenames are dogs and C are cats.
    # Keep the evaluation set unchanged for paired comparison.
    for name in ('train', 'calibration'):
        rows[name] = [r for r in rows[name] if r['group'].startswith('d')]
output = SERVER_ROOT / 'models' / datetime.now(timezone.utc).strftime('differential-%Y%m%d-%H%M%S')
output.mkdir()
model = load_screening_model(folder / 'classifier.h5')
base = next(layer for layer in model.layers if layer.name.startswith('efficientnet'))
base.trainable = False
print('extracting frozen features', flush=True)
features = {k: base.predict(image_dataset(v).map(lambda image, label: image), verbose=0) for k, v in rows.items() if k != 'test'}
labels = {k: np.array([CLASSES.index(r['disease']) for r in v]) for k, v in rows.items()}
def weights(name):
    counts = Counter(labels[name])
    values = np.array([len(labels[name]) / (len(CLASSES) * counts[int(y)]) for y in labels[name]], dtype=np.float32)
    if args.dog_breed_balanced and name == 'train':
        breed_counts = Counter((r['disease'], r['breed']) for r in rows[name])
        # Square-root smoothing and a cap avoid rare breeds dominating training.
        for disease in CLASSES:
            indices = [i for i, r in enumerate(rows[name]) if r['disease'] == disease]
            factors = np.array([min(3., 1 / np.sqrt(breed_counts[disease, rows[name][i]['breed']])) for i in indices])
            factors /= factors.mean()
            factors = np.clip(factors, .5, 3.)
            factors /= factors.mean()
            values[indices] *= factors
    return values
head = tf.keras.Sequential([tf.keras.layers.Input((1280,)), tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(.001)), tf.keras.layers.Dropout(.3), tf.keras.layers.Dense(5, activation='softmax')])
head.compile(optimizer=tf.keras.optimizers.Adam(.0005), loss='sparse_categorical_crossentropy')
print('stage 1: classifier head; all available training photos', len(rows['train']), flush=True)
history1 = head.fit(features['train'], labels['train'], sample_weight=weights('train'), validation_data=(features['calibration'], labels['calibration'], weights('calibration')), epochs=40, batch_size=64, verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'), tf.keras.layers.RandomRotation(.025, fill_mode='reflect'), tf.keras.layers.RandomContrast(.08)])
model = tf.keras.Model(inputs, head(base(augmentation(inputs), training=False)))
model.save(str(output / 'stage1.h5'), include_optimizer=False)
stage1_cal = model.predict(image_dataset(rows['calibration']), verbose=0)
score1 = classification_metrics(labels['calibration'], stage1_cal)['balanced_accuracy']
base.trainable = True
for layer in base.layers:
    layer.trainable = layer in base.layers[-35:] and not isinstance(layer, tf.keras.layers.BatchNormalization)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy')
print('stage 2: visual feature fine-tuning', flush=True)
train_data = image_dataset(rows['train'], True, True)
if args.dog_breed_balanced:
    # Attach breed-aware weights before shuffling so each weight stays with its image.
    paths = [r['image_path'] for r in rows['train']]
    def load(path, label, weight):
        image = tf.numpy_function(pixels_from_path, [path], tf.float32)
        image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
        return image, label, weight
    train_data = tf.data.Dataset.from_tensor_slices((paths, labels['train'], weights('train'))).shuffle(len(paths), seed=71).map(load, num_parallel_calls=4).batch(16).prefetch(2)
history2 = model.fit(train_data, validation_data=image_dataset(rows['calibration'], False, True), epochs=10, verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
stage2_cal = model.predict(image_dataset(rows['calibration']), verbose=0)
score2 = classification_metrics(labels['calibration'], stage2_cal)['balanced_accuracy']
if score1 > score2:
    model = load_screening_model(output / 'stage1.h5')
selected = 'head_only' if score1 > score2 else 'two_stage'
print('selection based only on calibration:', selected, score1, score2, flush=True)
probabilities = model.predict(image_dataset(rows['test']), verbose=0)
baseline = load_screening_model(folder / 'classifier.h5')
old_scores = baseline.predict(image_dataset(rows['test']), verbose=0)
metrics = classification_metrics(labels['test'], probabilities)
old_metrics = classification_metrics(labels['test'], old_scores)
model.save(str(output / 'classifier.h5'), include_optimizer=False)
restored = load_screening_model(output / 'classifier.h5')
images = np.stack([pixels_from_path(r['image_path']) for r in rows['test'][:10]])
error = float(np.abs(restored(images, training=False).numpy() - probabilities[:10]).max())
if error > 1e-4:
    raise RuntimeError('Export mismatch')
accepted = metrics['balanced_accuracy'] >= .6 and metrics['macro_f1'] >= .6 and min(metrics['recall_by_disease'].values()) >= .4
improved = metrics['accuracy'] > old_metrics['accuracy'] and metrics['balanced_accuracy'] > old_metrics['balanced_accuracy'] and metrics['macro_f1'] > old_metrics['macro_f1']
species_comparison = {}
for species in ('d', 'c'):
    mask = np.array([r['group'].startswith(species) for r in rows['test']])
    if mask.any():
        species_comparison[species] = {'samples': int(mask.sum()), 'previous_accuracy': float(np.mean(old_scores[mask].argmax(1) == labels['test'][mask])), 'candidate_accuracy': float(np.mean(probabilities[mask].argmax(1) == labels['test'][mask]))}
if args.dog_breed_balanced:
    improved = improved and all(v['candidate_accuracy'] >= v['previous_accuracy'] - .05 for v in species_comparison.values())
    improved = improved and all(metrics['recall_by_disease'][d] >= old_metrics['recall_by_disease'][d] - .05 for d in CLASSES)
report = {**warm, 'version': output.name, 'warm_start': previous, 'train_samples': len(rows['train']), 'calibration_samples': len(rows['calibration']), 'test': metrics, 'by_device': {'일반카메라': metrics}, 'accepted_for_reference_display': bool(accepted), 'experimental_forced_selection': not bool(accepted), 'architecture': 'head_training_then_last35_finetuning_rotation_contrast_flip', 'split_audit': audit, 'export_score_max_error': error, 'stages': {'head_epochs': len(history1.history['loss']), 'fine_tune_epochs': len(history2.history['loss']), 'selection': selected, 'calibration_balanced_accuracy': [score1, score2]}, 'paired_previous': old_metrics, 'improved_on_development_evaluation': bool(improved)}
(output / 'classifier.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
if args.dog_breed_balanced:
    report['training_scope'] = 'dogs_only_with_capped_breed_weights'
    report['species_comparison'] = species_comparison
    report['improved_on_development_evaluation'] = bool(improved)
    report['limitations'] = warm['limitations'] + ['Dog-only training; cat performance unsupported.']
    (output / 'classifier.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
np.savez(output / 'test-predictions.npz', labels=labels['test'], scores=probabilities, previous_scores=old_scores, paths=np.array([r['image_path'] for r in rows['test']]))
# Review samples are diagnostic only: never delete or relabel evaluation images.
wrong = np.flatnonzero(old_scores.argmax(1) != labels['test'])
sample = []
for disease in range(5):
    sample.extend([int(i) for i in wrong if labels['test'][i] == disease][:4])
sheet = Image.new('RGB', (800, len(sample) // 4 * 180), 'white')
draw = ImageDraw.Draw(sheet)
review = []
for position, i in enumerate(sample):
    r = rows['test'][i]
    with Image.open(r['image_path']) as photo:
        photo = photo.convert('RGB'); photo.thumbnail((195, 145))
        x, y = position % 4 * 200, position // 4 * 180
        sheet.paste(photo, (x, y))
        draw.text((x, y + 147), f'true={labels["test"][i]} old={old_scores[i].argmax()} new={probabilities[i].argmax()}', fill='black')
    annotation = None
    for annotation_path in Path(r['image_path']).parent.glob('*.json'):
        candidate = json.loads(annotation_path.read_text(encoding='utf-8-sig'))
        if candidate.get('label', {}).get('label_filename') == Path(r['image_path']).name:
            annotation = candidate
            break
    review.append({**r, 'annotation_found': annotation is not None, 'annotation_label': annotation.get('label', {}) if annotation else {}, 'old_prediction': CLASSES[old_scores[i].argmax()], 'new_prediction': CLASSES[probabilities[i].argmax()]})
sheet.save(output / 'errors-review.jpg')
(output / 'errors-review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding='utf-8')
if improved:
    pending = pointer.with_suffix('.pending.json'); pending.write_text(json.dumps({'version': output.name})); pending.replace(pointer)
print(json.dumps({'output': str(output), 'previous': old_metrics, 'candidate': metrics, 'published': bool(improved)}, ensure_ascii=False), flush=True)
