import os
import cv2
import numpy as np
from feature_extraction import extract_features
import sys

def build_database(dataset_dir):
    data = []
    image_paths = []

    for img_file in os.listdir(dataset_dir):
        img_path = os.path.join(dataset_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        features = extract_features(img)
        data.append(features)
        image_paths.append(img_path)

    np.save("features.npy", np.array(data))
    np.save("image_paths.npy", np.array(image_paths))
    print(f"Đã lưu {len(data)} ảnh vào CSDL.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("run python build_database.py <dataset_dir>")
    else:
        build_database(sys.argv[1])