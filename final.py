import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from model_wormhole import build_wormhole_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IMAGE_SIZE = 64
DATA_DIR = 'D:\\wormhole_detector\\data_test'
CATEGORIES = ['normal', 'blackhole', 'wormhole_candidate']
MODEL_WEIGHTS_PATH = 'temp_weights.h5'

def load_test_data():
    images, labels, file_names = [], [], []
    for idx, category in enumerate(CATEGORIES):
        path = os.path.join(DATA_DIR, category)
        for file in os.listdir(path):
            if file.endswith('.png'):
                img_path = os.path.join(path, file)
                img = Image.open(img_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
                images.append(np.array(img, dtype='float32') / 255.0)
                labels.append(idx)
                file_names.append(file)
    images = np.expand_dims(np.array(images), axis=-1)
    return images, np.array(labels), file_names

print("Loading data...")
X_test, y_true, filenames = load_test_data()
print(f"Loaded: {len(y_true)} test samples")

model = build_wormhole_model(input_shape=(64, 64, 1), num_classes=len(CATEGORIES))
model.load_weights(MODEL_WEIGHTS_PATH)
print("Model weights loaded")

preds = model.predict(X_test)
y_pred = np.argmax(preds, axis=1)

print(f"\nTest Accuracy: {accuracy_score(y_true, y_pred):.4f} | Classification Report:")
print(classification_report(y_true, y_pred, target_names=CATEGORIES))

print("\nSample Predictions:")
for i in range(min(10, len(filenames))):
    print(f"{filenames[i]} → True: {CATEGORIES[y_true[i]]} | Predicted: {CATEGORIES[y_pred[i]]}")

def visualize_prediction(img_array, prediction_label, confidence):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_array.squeeze(), cmap='gray')
    ax.set_title(f"Prediction: {prediction_label} ({confidence:.2f})")
    ax.axis('off')
    if prediction_label == "wormhole_candidate":
        y, x = np.indices((64, 64))
        cx, cy = 32, 32
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        ripple = 0.5 * np.sin(0.4 * r + confidence * 8)
        ax.imshow(ripple, cmap='inferno', alpha=0.6)
        ax.text(2, 60, 'Wormhole Signature Detected', color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
    plt.tight_layout()
    plt.show()

def show_realistic_wormhole():
    if not os.path.exists("realistic_wormhole.png"):
        print("realistic_wormhole.png not found!")
        return
    img = plt.imread("realistic_wormhole.png")
    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    plt.title("Ultra-Realistic Wormhole Visualization", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

for idx in range(len(X_test)):
    sample_img = X_test[idx]
    sample_pred = model.predict(np.expand_dims(sample_img, axis=0))
    pred_label = np.argmax(sample_pred)
    conf = np.max(sample_pred)
    pred_name = CATEGORIES[pred_label]
    if pred_name == "wormhole_candidate":
        print(f"\nVisualizing Wormhole Candidate: {filenames[idx]} | Confidence: {conf:.2f}")
        visualize_prediction(sample_img, pred_name, conf)
        show_realistic_wormhole()
        break
else:
    print("\nNo wormhole candidates found in sample test set.")
