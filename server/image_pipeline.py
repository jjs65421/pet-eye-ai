"""Identical RGB, orientation and bilinear 0..255 preprocessing everywhere."""
import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = 224
PREPROCESSING = "exif-rgb-bilinear-224-float32-0-255-v1"


def pixels_from_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR), dtype=np.float32)


def pixels_from_path(path):
    if isinstance(path, np.ndarray):
        path = path.item()
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    with Image.open(path) as image:
        return pixels_from_image(image)
