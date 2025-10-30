TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

PLOTS_DIR = "plots"
MODEL_DIR = "saved_models"
MODEL_NAME = "chest_xray_model.h5"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 25

INPUT_SHAPE = (64, 64, 3)
DROPOUT_CONV = 0.2
DROPOUT_DENSE = 0.4


EARLY_STOPPING_PATIENCE = 5