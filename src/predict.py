import numpy as np

def get_predictions(model, test_flow):
    y_pred_prob = model.predict(test_flow).reshape(-1)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    y_true = test_flow.classes
    return y_true, y_pred, y_pred_prob