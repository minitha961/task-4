import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load images from folder
def load_images(folder, label):
    data = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠ Warning: Skipped unreadable image {filename}")
                continue
            img = cv2.resize(img, (64, 64))  # Resize all to same size
            img = img.flatten() / 255.0     # Normalize
            data.append((img, label))
    return data

# Load real and fake images
real_data = load_images('dataset/R', 1)   # 1 = Real
fake_data = load_images('dataset/F', 0)   # 0 = Fake

# Combine & prepare dataset
all_data = real_data + fake_data
X = np.array([i[0] for i in all_data])
y = np.array([i[1] for i in all_data])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
import joblib
joblib.dump(model, 'model.pkl')  # Save the model


# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Predict a single test image
test_image_path = 'dataset/R/him.jpg'  # Replace with your actual image if needed
sample = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

if sample is None:
    print(f"❌ Error: Could not load image '{test_image_path}'")
else:
    sample = cv2.resize(sample, (64, 64)).flatten().reshape(1, -1) / 255.0
    prediction = model.predict(sample)[0]
    print("🧠 Prediction for test image:", "Real" if prediction == 1 else "Fake")