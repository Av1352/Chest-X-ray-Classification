import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
import os

def plot_metrics(history, plots_dir):
    plt.figure()
    plt.plot(history.history['loss'], label="Train loss")
    plt.plot(history.history['val_loss'], label="Val loss")
    plt.legend(), plt.title("Loss"), plt.xlabel("Epoch"), plt.ylabel("Loss")
    plt.savefig(os.path.join(plots_dir, "train_val_loss.jpg")), plt.close()

    plt.figure()
    plt.plot(history.history['accuracy'], label="Train Acc")
    plt.plot(history.history['val_accuracy'], label="Val Acc")
    plt.legend(), plt.title("Accuracy"), plt.xlabel("Epoch"), plt.ylabel("Acc")
    plt.savefig(os.path.join(plots_dir, "train_val_accuracy.jpg")), plt.close()

def plot_confusion_roc(y_true, y_pred, y_pred_prob, plots_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pneumonia"])
    disp.plot(cmap="viridis", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.jpg")), plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR"), plt.ylabel("TPR"), plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "roc_curve.jpg")), plt.close()

def print_classification_report(y_true, y_pred):
    report_text = classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"])
    print("\nClassification Report:\n", report_text)
    with open("plots/report.txt", "w") as f:
        f.write(report_text)
