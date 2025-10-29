import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf

IMG_SIZE = (224, 224)   
BATCH_SIZE = 32
EPOCHS = 15
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
MODEL_SAVE_PATH = "saved_models/chest_xray_model.h5"

def main():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=12,
        width_shift_range=0.08,
        height_shift_range=0.08,
        brightness_range=[0.8,1.2],
        shear_range=0.2,
        zoom_range=0.12,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load from folders with flow_from_directory!
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    print("Class mapping:", train_gen.class_indices)

    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
    ]

    print("Training model...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print(f"Model saved at: {MODEL_SAVE_PATH}")

    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "train_val_accuracy.png"))
    plt.close()

    # Save loss plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "train_val_loss.png"))
    plt.close()


if __name__ == "__main__":
    main()
