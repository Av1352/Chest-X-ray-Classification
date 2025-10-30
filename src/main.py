# src/main.py

from src.data_loader import load_images_and_labels
from src.preprocess import preprocess_data
from src.model import build_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

IMG_SIZE = (64, 64)
DATA_DIR = "data/train"  # Adjust if needed
MODEL_SAVE_PATH = "saved_models/classic_chest_xray_model.h5"

# 1. Load and preprocess all training images
print("Loading images...")
images, labels = load_images_and_labels(DATA_DIR, target_size=IMG_SIZE)
print(f"Loaded {len(images)} images.")

# 2. Split and normalize
print("Preprocessing and splitting data...")
X_train, X_test, y_train, y_test = preprocess_data(images, labels, test_size=0.2)

# 3. Build and train model
print("Building and training model...")
model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32,
    verbose=2
)

# 4. Save model
model.save(MODEL_SAVE_PATH)
print(f"Model trained and saved to {MODEL_SAVE_PATH}")

# 5. Evaluate numeric metrics
score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {score[0]:.4f}, Test accuracy: {score[1]:.4f}")

# 6. Generate and save plots
# Loss curve
plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('train_val_loss.jpg')
plt.close()

# Accuracy curve
plt.figure()
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('train_val_accuracy.jpg')
plt.close()

# Confusion Matrix
y_pred_prob = model.predict(X_test).reshape(-1)
y_pred = (y_pred_prob >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pneumonia"])
disp.plot(cmap='viridis', values_format='d')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.jpg")
plt.close()

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.jpg')
plt.close()

print("All plots saved (train_val_loss.jpg, train_val_accuracy.jpg, confusion_matrix.jpg, roc_curve.jpg)")