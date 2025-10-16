import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Configuration
image_size = (64, 64)
data_dir = "dataset"

# Extract color histogram (better than raw pixels)
def extract_color_histogram(image, bins=(8, 8, 8)):
    img = np.array(image)
    hist = np.histogramdd(
        img.reshape(-1, 3),
        bins=bins,
        range=[(0, 256), (0, 256), (0, 256)]
    )[0]
    hist = hist.flatten()
    return hist / hist.sum()

# Load and process images from dataset folder
def load_images(data_dir):
    X, y = [], []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            try:
                img_path = os.path.join(label_path, file)
                img = Image.open(img_path).convert('RGB').resize(image_size)
                features = extract_color_histogram(img)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Skipped {file}: {e}")
                continue
    return np.array(X), np.array(y)

# Load dataset
X, y = load_images(data_dir)

# Print debug info
print(f"Classes found: {np.unique(y)}")
print(f"Samples per class:")
for label in np.unique(y):
    print(f"  {label}: {(y == label).sum()} images")

# Split data (stratified = balanced class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train k-NN classifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}")

# Predict user-provided image
def predict_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB').resize(image_size)
        features = extract_color_histogram(img).reshape(1, -1)
        prediction = model.predict(features)
        print(f"🔍 Prediction: {prediction[0]}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")

# Ask for user input
img_path = input("\n📸 Enter image path to classify: ").strip()
predict_image(img_path)
