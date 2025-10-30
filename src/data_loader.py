import os
from PIL import Image
import numpy as np

def load_images_and_labels(data_dir, target_size=(64, 64)):
    images = []
    labels = []
    class_map = {}
    classes = sorted(os.listdir(data_dir))
    for idx, cls in enumerate(classes):
        class_map[idx] = cls
        class_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB").resize(target_size)
                images.append(np.array(img, dtype=np.uint8))
                labels.append(idx)
            except Exception:
                continue
    return np.array(images, dtype=np.uint8), np.array(labels), class_map