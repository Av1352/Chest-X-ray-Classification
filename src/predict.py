from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

def predict_image(model, img, target_size=(224,224)):
    img = img.resize(target_size)
    img = np.expand_dims(np.array(img)/255.0, axis=0)
    pred = model.predict(img)[0][0]
    label = "Pneumonia" if pred > 0.5 else "Normal"
    return label, float(pred)