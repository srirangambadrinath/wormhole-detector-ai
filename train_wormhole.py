import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from model_wormhole import build_wormhole_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IMAGE_SIZE = 64
DATA_DIR = 'D:\\wormhole_detector\\data_test'
CATEGORIES = ['normal', 'blackhole', 'wormhole_candidate']
EPOCHS = 15
BATCH_SIZE = 32

def load_data():
    images = []
    labels = []
    for idx, category in enumerate(CATEGORIES):
        path = os.path.join(DATA_DIR, category)
        for file in os.listdir(path):
            if file.endswith('.png'):
                img_path = os.path.join(path, file)
                img = Image.open(img_path).convert('L')
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img_array = np.array(img, dtype='float32') / 255.0
                images.append(img_array)
                labels.append(idx)
    images = np.expand_dims(np.array(images), axis=-1)
    labels = to_categorical(np.array(labels), num_classes=len(CATEGORIES))
    return images, labels

print("Loading data...")
X, y = load_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Loaded: {X_train.shape[0]} train, {X_val.shape[0]} val samples")

print("X_train shape:", X_train.shape, "| dtype:", X_train.dtype)
print("y_train shape:", y_train.shape, "| dtype:", y_train.dtype)
print("Sample y_train (first 5):\n", y_train[:5])
print("Sample label indices:", np.argmax(y_train[:5], axis=1))

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')

model = build_wormhole_model(input_shape=(64, 64, 1), num_classes=len(CATEGORIES))
model.summary()

print("Starting model training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)
print("Model training complete.")

model.save_weights("temp_weights.h5")
print("Model weights saved to temp_weights.h5")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()
