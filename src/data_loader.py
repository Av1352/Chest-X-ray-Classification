import os
import cv2
import numpy as np

def load_data(data_dir: str, target_size=(224, 224)):
    images, labels = [], []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, target_size)
            images.append(img)
            labels.append(0 if label == "NORMAL" else 1)
    return np.array(images), np.array(labels)