from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import numpy as np

def evaluate_model(model_path, X_test, y_test):
    model = load_model(model_path)
    y_pred_probs = model.predict(X_test).flatten()
    y_pred = (y_pred_probs > 0.5).astype('int')
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)
    cm = confusion_matrix(y_test, y_pred)
    return {"accuracy": acc, "auc": auc, "confusion_matrix": cm}