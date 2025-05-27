import cv2
import numpy as np
from skimage.feature import hog

def extract_color_histogram(img, bins=(8, 12, 3)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_hog(img, resize_dim=(128, 128)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resize_dim)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    return features

def extract_features(img, use_hist=True, use_hog=True):
    features = []
    if use_hist:
        features.extend(extract_color_histogram(img))
    if use_hog:
        features.extend(extract_hog(img))
    return np.array(features)