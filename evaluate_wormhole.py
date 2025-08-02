import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import to_categorical
from model_wormhole import build_wormhole_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IMAGE_SIZE = 64
DATA_DIR = 'D:\\wormhole_detector\\data_test'
CATEGORIES = ['normal', 'blackhole', 'wormhole_candidate']
MODEL_WEIGHTS_PATH = 'temp_weights.h5'

def load_test_data():
    images = []
    labels = []
    file_names = []
    
    for idx, category in enumerate(CATEGORIES):
        path = os.path.join(DATA_DIR, category)
        for file in os.listdir(path):
            if file.endswith('.png'):
                img_path = os.path.join(path, file)
                img = Image.open(img_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
                img_array = np.array(img, dtype='float32') / 255.0
                images.append(img_array)
                labels.append(idx)
                file_names.append(file)

    images = np.expand_dims(np.array(images), axis=-1)
    labels = to_categorical(np.array(labels), num_classes=len(CATEGORIES))
    return images, labels, file_names

print("Loading data...")
X_test, y_test, file_names = load_test_data()
print(f"Loaded: {X_test.shape[0]} test samples")

model = build_wormhole_model(input_shape=(64, 64, 1), num_classes=len(CATEGORIES))
model.load_weights(MODEL_WEIGHTS_PATH)
print("Model weights loaded")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f} | Loss: {loss:.4f}")

y_pred = model.predict(X_test)
predicted_classes = np.argmax(y_pred, axis=1)
true_classes = np.argmax(y_test, axis=1)

print("\nSample Predictions:")
for i in range(min(10, len(file_names))):
    print(f"{file_names[i]} → True: {CATEGORIES[true_classes[i]]} | Predicted: {CATEGORIES[predicted_classes[i]]}")
