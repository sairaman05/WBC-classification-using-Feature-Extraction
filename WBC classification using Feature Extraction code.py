import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# Step 1: Load and preprocess the dataset
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if not os.path.isdir(label_path):
            continue
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (128, 128))
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

# Step 2: Watershed Segmentation for WBC extraction
def extract_wbc(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Use Otsu's thresholding to create a binary image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Perform morphological operations to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area using distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Unknown region (subtract sure foreground from sure background)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1  # Add 1 to all labels so that sure background is 1, not 0
    markers[unknown == 255] = 0  # Mark the unknown region as 0
    
    # Apply watershed algorithm
    markers = cv2.watershed(image, markers)
    
    # Create a mask for the WBC
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers > 1] = 255  # WBC region is marked with values > 1
    
    # Extract the WBC using the mask and set background to black
    wbc = cv2.bitwise_and(image, image, mask=mask)
    wbc[mask == 0] = [0, 0, 0]  # Set pixels outside the mask to black
    
    return wbc, mask

# Step 3: Display segmented WBC images (one from each class)
def display_segmented_wbc_images(images, labels):
    class_images = {}
    for image, label in zip(images, labels):
        if label not in class_images:
            wbc, _ = extract_wbc(image)
            class_images[label] = wbc
        if len(class_images) == 5:
            break
    
    plt.figure(figsize=(15, 10))
    for i, (label, wbc) in enumerate(class_images.items()):
        plt.subplot(1, 5, i + 1)
        plt.imshow(cv2.cvtColor(wbc, cv2.COLOR_BGR2RGB))
        plt.title(f"Class: {label}")
        plt.axis("off")
    plt.show()

# Step 4: Feature extraction using a pre-trained CNN (VGG16)
def extract_cnn_features(images):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)
    features = model.predict(images)
    features = features.reshape(features.shape[0], -1)
    return features

# Step 5: Fixed Handcrafted Feature Extraction
def extract_handcrafted_features(images):
    features = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding instead of global thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Find the largest contour (assumed to be the WBC)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Compute features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Additional features: Aspect Ratio and Extent
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h
            rect_area = w * h
            extent = float(area) / rect_area
            
            features.append([area, perimeter, solidity, aspect_ratio, extent])
        else:
            features.append([0, 0, 0, 0, 0])
    return np.array(features)

# Load train and test datasets
train_images, train_labels = load_images_from_folder(r"/Users/saikrishnaa/Downloads/archive/train")
test_images, test_labels = load_images_from_folder(r"/Users/saikrishnaa/Downloads/archive/test-A")

# Display 5 segmented WBC images (one from each class)
display_segmented_wbc_images(train_images, train_labels)

# Extract WBC from all training and test images
train_wbc_images = [extract_wbc(image)[0] for image in train_images]
test_wbc_images = [extract_wbc(image)[0] for image in test_images]

# Extract CNN features
train_cnn_features = extract_cnn_features(np.array(train_wbc_images))
test_cnn_features = extract_cnn_features(np.array(test_wbc_images))

# Extract handcrafted features
train_handcrafted_features = extract_handcrafted_features(train_wbc_images)
test_handcrafted_features = extract_handcrafted_features(test_wbc_images)

# Print handcrafted features for the first 5 images
print("Handcrafted Features for the first 5 images:")
for i, features in enumerate(train_handcrafted_features[:5]):
    print(f"Image {i + 1}: Area={features[0]:.2f}, Perimeter={features[1]:.2f}, Solidity={features[2]:.2f}, "
          f"Aspect Ratio={features[3]:.2f}, Extent={features[4]:.2f}")

# Step 6: Combine CNN features and handcrafted features
train_features = np.hstack((train_cnn_features, train_handcrafted_features))
test_features = np.hstack((test_cnn_features, test_handcrafted_features))

# Step 7: Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
train_features_balanced, train_labels_balanced = smote.fit_resample(train_features, train_labels)

# Step 8: Train a LightGBM classifier
label_encoder = LabelEncoder()
train_labels_balanced_encoded = label_encoder.fit_transform(train_labels_balanced)
test_labels_encoded = label_encoder.transform(test_labels)

train_data = lgb.Dataset(train_features_balanced, label=train_labels_balanced_encoded)

params = {
    'objective': 'multiclass',
    'num_class': 5,
    'boosting_type': 'gbdt',
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
}

clf = lgb.train(params, train_data, num_boost_round=100)

# Step 9: Evaluate the model
y_pred = clf.predict(test_features)
y_pred_classes = np.argmax(y_pred, axis=1)
print("Accuracy:", accuracy_score(test_labels_encoded, y_pred_classes))
print("Classification Report:\n", classification_report(test_labels_encoded, y_pred_classes))

# Save the model (optional)
clf.save_model("wbc_classification_lightgbm_model.txt")
