from sklearn.model_selection import train_test_split

def preprocess_data(images, labels):
    images = images / 255.0
    return train_test_split(images, labels, test_size=0.2, random_state=42)