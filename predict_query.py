import cv2
import numpy as np
from feature_extraction import extract_features
import matplotlib.pyplot as plt

def show_results(query_img, neighbors, distances, label_map, y_train, img_paths):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Ảnh truy vấn")
    axs[0].axis("off")

    for i in range(3):
        neighbor_img = cv2.imread(img_paths[neighbors[i]])
        axs[i + 1].imshow(cv2.cvtColor(neighbor_img, cv2.COLOR_BGR2RGB))
        label = label_map[y_train[neighbors[i]]]
        axs[i + 1].set_title(f"Top {i+1}: {label}\nKhoảng cách: {distances[i]:.2f}")
        axs[i + 1].axis("off")
    plt.tight_layout()
    plt.show()

def predict(image_path, k=3):
    X_train = np.load("bird_features.npy", allow_pickle=True)
    y_train = np.load("bird_labels.npy", allow_pickle=True)
    label_map = np.load("bird_label_map.npy", allow_pickle=True).item()
    img_paths = np.load("bird_image_paths.npy", allow_pickle=True)

    query_img = cv2.imread(image_path)
    query_feat = extract_features(query_img).reshape(1, -1)

    distances = np.linalg.norm(X_train - query_feat, axis=1)
    top_k_indices = np.argsort(distances)[:k]

    print("Top 3 ảnh giống nhất:")
    for rank, idx in enumerate(top_k_indices):
        label = label_map[y_train[idx]]
        dist = distances[idx]
        print(f"Top {rank+1}: {label} - Khoảng cách: {dist:.2f}")

    show_results(query_img, top_k_indices, distances[top_k_indices], label_map, y_train, img_paths)