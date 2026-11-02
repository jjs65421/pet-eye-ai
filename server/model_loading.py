"""Restore preprocessing state after HDF5 loading in the existing TF 2.7 runtime."""
import tensorflow as tf


def load_screening_model(path):
    model = tf.keras.models.load_model(str(path), compile=False)
    # TF 2.7's Normalization stores adapt_mean/variance as weights, but the
    # tensors consumed by call() can retain build-time values after HDF5 loading.
    # Rebuild those tensors from the restored weights, including nested models.
    for layer in model.submodules:
        if isinstance(layer, tf.keras.layers.Normalization):
            layer.finalize_state()
    return model
