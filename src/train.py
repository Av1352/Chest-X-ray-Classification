import os
from src.data_loader import load_images_and_labels
from src.preprocess import preprocess_data
from src.model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

PLOTS_DIR = "plots"
MODEL_DIR = "saved_models"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_DIR = "data/train"
VAL_DIR = "data/val"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

images, labels, _ = load_images_and_labels(DATA_DIR, target_size=IMG_SIZE)
val_images, val_labels, _ = load_images_and_labels(VAL_DIR, target_size=IMG_SIZE)
X_train = images.astype('float32') / 255.0
y_train = labels
X_val = val_images.astype('float32') / 255.0
y_val = val_labels

model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

cb = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(os.path.join(MODEL_DIR, "classic_chest_xray_model.h5"), save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=BATCH_SIZE,
    callbacks=cb
)

# Save plots
plt.figure()
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.legend(), plt.title("Loss"), plt.xlabel("Epoch"), plt.ylabel("Loss")
plt.savefig(os.path.join(PLOTS_DIR, "train_val_loss.jpg"))
plt.close()
plt.figure()
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.legend(), plt.title("Accuracy"), plt.xlabel("Epoch"), plt.ylabel("Accuracy")
plt.savefig(os.path.join(PLOTS_DIR, "train_val_accuracy.jpg"))
plt.close()