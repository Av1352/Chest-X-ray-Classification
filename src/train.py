from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config import EPOCHS, EARLY_STOPPING_PATIENCE

def train_model(model, train_flow, val_flow, model_path):
    callbacks = [
        EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, save_best_only=True, verbose=1)
    ]
    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    return history