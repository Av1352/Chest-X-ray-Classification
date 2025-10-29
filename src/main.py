import os
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_images_and_labels
from src.preprocess import preprocess_data
from src.model import build_model
from src.evaluate import evaluate_model

IMG_SIZE = (224, 224)
DATA_DIR = "data/train"
MODEL_SAVE_PATH = "saved_models/chest_xray_model.h5"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

images, labels = load_images_and_labels(DATA_DIR, target_size=IMG_SIZE)
X_train, X_test, y_train, y_test = preprocess_data(images, labels, test_size=0.2)

model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15, batch_size=32,
    verbose=2
)
model.save(MODEL_SAVE_PATH)

plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "train_val_accuracy.png"))
plt.close()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "train_val_loss.png"))
plt.close()

metrics = evaluate_model(MODEL_SAVE_PATH, X_test, y_test)

from sklearn.metrics import roc_curve
y_pred_probs = model.predict(X_test).flatten()
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve (AUC={:.2f})".format(metrics["auc"]))
plt.title("ROC Curve")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"))
plt.close()

from sklearn.metrics import ConfusionMatrixDisplay
cm = metrics["confusion_matrix"]
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
plt.close()

print("Model trained, evaluated, and all plots saved to ./plots/")