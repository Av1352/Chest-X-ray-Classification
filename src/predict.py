from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

def predict_image(model_path, img, target_size=(224,224)):
    model = load_model(model_path)
    if isinstance(img, Image.Image):
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    else:
        img_array = np.array(img).reshape((1,)+target_size+(3,)) / 255.0
    pred = model.predict(img_array)[0][0]
    label = "Pneumonia" if pred > 0.5 else "Normal"
    return label, float(pred)
