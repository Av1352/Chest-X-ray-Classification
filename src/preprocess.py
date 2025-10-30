from sklearn.model_selection import train_test_split
import numpy as np

def preprocess_data(images, labels, test_size=0.2, random_state=42):
    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return X_train, X_test, y_train, y_test