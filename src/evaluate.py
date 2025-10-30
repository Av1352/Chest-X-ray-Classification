from src.data_loader import load_images_and_labels
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import numpy as np
import os
import matplotlib.pyplot as plt

def evaluate(model_path, data_dir, plots_dir="plots"):
    os.makedirs(plots_dir, exist_ok=True)
    images, labels, class_map = load_images_and_labels(data_dir, target_size=(64, 64))
    x = np.array(images, dtype=np.float32) / 255.0
    y_true = np.array(labels)
    model = load_model(model_path)
    y_pred_prob = model.predict(x).reshape(-1)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"])
    print("\nClassification Report:\n", report)
    with open(os.path.join(plots_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pneumonia"])
    disp.plot(cmap='viridis', values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.jpg"))
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR"), plt.ylabel("TPR"), plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "roc_curve.jpg"))
    plt.close()
