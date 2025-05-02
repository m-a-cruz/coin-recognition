from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
import os, cv2, numpy as np
import joblib

X, y = [], []
labels = os.listdir('new_coins')

for label in labels:
    for fname in os.listdir(f'new_coins/{label}'):
        img = cv2.imread(f'new_coins/{label}/{fname}', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))  # Resize for consistency
        features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        X.append(features)
        y.append(label)

clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X, y)

joblib.dump(clf, "coin_classifier.pkl")


from skimage.feature import hog
from joblib import load
import cv2

clf = load("coin_classifier.pkl")  # Load trained model

def predict_coin_label(cropped_img):
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    features = hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    label = clf.predict([features])[0]
    confidence = round(clf.predict_proba([features])[0].max(), 2)
    return label, confidence


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('test/5.jpg')
if image is None:
    raise FileNotFoundError("Image not loaded.")

# Resize for faster processing
scale_percent = 70  # Adjust if needed
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur
blurred = cv2.GaussianBlur(gray, (7, 7), 1)


# Use Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)


for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 50: 
        continue

    # Calculate circularity
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    if circularity < 0.1:  # filter non-circular objects
        continue

    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    x, y, w, h = cv2.boundingRect(cnt)
    coin_crop = image[y:y+h, x:x+w]

    label, confidence = predict_coin_label(coin_crop)

    cv2.circle(output_img, center, radius, (0, 255, 0), 1)
    cv2.putText(output_img, f"P{label}", (center[0]-10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)

# # Show result
plt.figure(figsize=(10, 6))
plt.imshow(output_img)
plt.title('Detected and Classified Coins')
plt.axis('off')
plt.show()

cv2.imwrite('FP5.jpg', output_img)

