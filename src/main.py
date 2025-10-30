import os
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, PLOTS_DIR, MODEL_DIR, MODEL_NAME
from preprocess import create_dirs
from data_loader import get_data_generators
from model import build_model
from train import train_model
from predict import get_predictions
from evaluate import plot_metrics, plot_confusion_roc, print_classification_report

# Paths
model_path = os.path.join(MODEL_DIR, MODEL_NAME)

# Create directories
create_dirs(PLOTS_DIR, MODEL_DIR)

# Load data
train_flow, val_flow, test_flow = get_data_generators(TRAIN_DIR, VAL_DIR, TEST_DIR)

# Build model
model = build_model()

# Train
history = train_model(model, train_flow, val_flow, model_path)

# Evaluate on test set
score = model.evaluate(test_flow)
print(f"Test loss: {score[0]:.4f}, Test accuracy: {score[1]:.4f}")

# Plot metrics
plot_metrics(history, PLOTS_DIR)

# Predictions
y_true, y_pred, y_pred_prob = get_predictions(model, test_flow)

# Confusion & ROC
plot_confusion_roc(y_true, y_pred, y_pred_prob, PLOTS_DIR)

print_classification_report(y_true, y_pred)

print("Plots saved to ./plots and model saved to ./saved_models!")