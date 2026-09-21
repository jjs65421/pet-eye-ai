from pathlib import Path

import numpy as np
import tensorflow as tf

from train import DISEASE_ALIASES, SMOKE_DATA_ROOT, dataset, samples_for


model_path = Path(__file__).parent / "models" / "백내장.keras"
paths, targets = samples_for(SMOKE_DATA_ROOT / "validation" / "labels", DISEASE_ALIASES["cataract"])
model = tf.keras.models.load_model(model_path)
probabilities = model.predict(dataset(paths, targets, False), verbose=0).reshape(-1)
predictions = (probabilities >= 0.5).astype(int)
accuracy = float(np.mean(predictions == np.asarray(targets)))
print(f"validation_samples={len(targets)}")
print(f"validation_accuracy={accuracy:.4f}")
print(f"mean_probability={float(np.mean(probabilities)):.4f}")
