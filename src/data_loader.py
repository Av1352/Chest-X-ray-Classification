import os
from PIL import Image
import numpy as np

def load_images_and_labels(data_dir, target_size=(224, 224)):
    images = []
    labels = []
    label_map = {'NORMAL': 0, 'PNEUMONIA': 1}
    for label_name, label_val in label_map.items():
        class_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, fname)
                img = Image.open(img_path).convert("RGB").resize(target_size)
                images.append(np.array(img))
                labels.append(label_val)
    return np.array(images), np.array(labels)