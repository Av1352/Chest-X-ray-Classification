import os
import random
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

MODEL_PATH = "saved_models/chest_xray_model.h5"
IMG_SIZE = (64, 64)
N = 20

def get_random_samples(class_dir, n):
    files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg') or f.lower().endswith('.png')]
    return random.sample(files, min(n, len(files)))

normal_dir = "data/test/NORMAL"
pneumonia_dir = "data/test/PNEUMONIA"

normal_samples = get_random_samples(normal_dir, N // 2)
pneumonia_samples = get_random_samples(pneumonia_dir, N // 2)

model = load_model(MODEL_PATH)

correct = 0
wrong = 0
details = []

for img_path in normal_samples:
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr)[0][0])
    pred = "Pneumonia" if prob >= 0.5 else "Normal"
    is_correct = pred == "Normal"
    correct += int(is_correct)
    wrong += int(not is_correct)
    details.append((img_path, "Normal", prob, pred, is_correct))

for img_path in pneumonia_samples:
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr)[0][0])
    pred = "Pneumonia" if prob >= 0.5 else "Normal"
    is_correct = pred == "Pneumonia"
    correct += int(is_correct)
    wrong += int(not is_correct)
    details.append((img_path, "Pneumonia", prob, pred, is_correct))

print(f"\nTested {N} random images (Normal: {len(normal_samples)}, Pneumonia: {len(pneumonia_samples)})")
print(f"Correct: {correct}, Wrong: {wrong}, Accuracy: {correct / N:.2f}\n")

for img_path, true_label, prob, pred, is_correct in details:
    flag = "CORRECT" if is_correct else "WRONG"
    print(f"Image: {img_path}\nTrue: {true_label}, Raw Prob: {prob:.4f}, Predicted: {pred} --> {flag}\n")
