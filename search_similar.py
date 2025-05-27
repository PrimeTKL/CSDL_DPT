import cv2
import numpy as np
from feature_extraction import extract_features
import matplotlib.pyplot as plt
import sys

def show_results(query_img, top_indices, distances, image_paths):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Ảnh truy vấn")
    axs[0].axis("off")

    for i in range(3):
        img = cv2.imread(image_paths[top_indices[i]])
        axs[i + 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i + 1].set_title(f"Top {i+1}\nKhoảng cách: {distances[i]:.2f}")
        axs[i + 1].axis("off")

    plt.tight_layout()
    plt.show()

def search(query_img_path):
    X = np.load("features.npy", allow_pickle=True)
    img_paths = np.load("image_paths.npy", allow_pickle=True)

    query_img = cv2.imread(query_img_path)
    query_feat = extract_features(query_img).reshape(1, -1)

    distances = np.linalg.norm(X - query_feat, axis=1)
    top_k = np.argsort(distances)[:3]

    print("Top 3 ảnh giống nhất:")
    for i, idx in enumerate(top_k):
        print(f"Top {i+1}: {img_paths[idx]} - Khoảng cách: {distances[idx]:.2f}")

    show_results(query_img, top_k, distances[top_k], img_paths)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python search_similar.py <query_image_path>")
    else:
        search(sys.argv[1])