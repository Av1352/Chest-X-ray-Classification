from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from config import INPUT_SHAPE, DROPOUT_CONV, DROPOUT_DENSE

def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=INPUT_SHAPE),
        MaxPooling2D(2,2),
        Dropout(DROPOUT_CONV),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Dropout(DROPOUT_CONV),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(DROPOUT_DENSE),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model