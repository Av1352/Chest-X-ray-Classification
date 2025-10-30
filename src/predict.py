import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

def predict_image(model_path, img_path, target_size=(64, 64)):
    model = load_model(model_path)
    img = Image.open(img_path).convert("RGB").resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr)[0][0])
    return prob, "Pneumonia" if prob >= 0.5 else "Normal"