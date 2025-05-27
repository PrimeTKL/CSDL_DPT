import cv2
import numpy as np
import os
from feature_extraction import extract_features  # Đảm bảo bạn đã lưu đoạn extract ở file này

# Thư mục chứa ảnh
image_dir = "dataset_images/"
# Thư mục lưu vector đặc trưng
feature_dir = "features/"

# Tạo thư mục nếu chưa có
os.makedirs(feature_dir, exist_ok=True)

# Duyệt tất cả ảnh trong thư mục
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)

        # Trích xuất vector đặc trưng
        features = extract_features(img)

        # Tên file đặc trưng tương ứng
        feature_filename = os.path.splitext(filename)[0] + ".npy"
        feature_path = os.path.join(feature_dir, feature_filename)

        # Lưu vector đặc trưng
        np.save(feature_path, features)
        print(f"Đã lưu vector đặc trưng: {feature_path}")
