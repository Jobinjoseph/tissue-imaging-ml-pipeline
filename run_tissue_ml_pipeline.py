# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 07:57:03 2025

@author: joseph
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 1. Load image and preprocess
def load_and_preprocess(image_path, patch_size=128):
    # Read image (BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image {image_path} not found.")
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixels to [0,1]
    img = img / 255.0
    
    # Crop or pad image to be divisible by patch_size
    h, w, _ = img.shape
    h_cropped = (h // patch_size) * patch_size
    w_cropped = (w // patch_size) * patch_size
    img = img[:h_cropped, :w_cropped, :]
    
    # Split into patches
    patches = []
    for i in range(0, h_cropped, patch_size):
        for j in range(0, w_cropped, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
    
    patches = np.array(patches)
    print(f"Created {len(patches)} patches of size {patch_size}x{patch_size}")
    return patches

# 2. Simple CNN model for patch classification
def create_model(input_shape=(128,128,3), num_classes=2):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. Demo training with synthetic labels
def train_demo(patches, epochs=3):
    num_patches = patches.shape[0]
    
    # Create synthetic labels: randomly assign 0 or 1
    labels = np.random.randint(0, 2, size=num_patches)
    
    # One-hot encode
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=2)
    
    model = create_model()
    model.fit(patches, labels_onehot, epochs=epochs, batch_size=32)
    return model

# 4. Predict on patches and save heatmap
def predict_and_visualize(model, patches, original_image_path, patch_size=128):
    preds = model.predict(patches)
    classes = np.argmax(preds, axis=1)
    
    # Load original image for visualization
    orig = cv2.imread(original_image_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    h, w, _ = orig.shape
    h_cropped = (h // patch_size) * patch_size
    w_cropped = (w // patch_size) * patch_size
    
    heatmap = np.zeros((h_cropped, w_cropped))
    
    idx = 0
    for i in range(0, h_cropped, patch_size):
        for j in range(0, w_cropped, patch_size):
            heatmap[i:i+patch_size, j:j+patch_size] = classes[idx]
            idx += 1
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original Image (cropped)")
    plt.imshow(orig[:h_cropped, :w_cropped])
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.title("Predicted Classes Heatmap")
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    image_path = "path_to_your_tissue_image.jpg"  # Replace with your image path
    
    patches = load_and_preprocess(image_path)
    model = train_demo(patches)
    predict_and_visualize(model, patches, image_path)
