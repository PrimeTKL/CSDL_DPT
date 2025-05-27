import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from feature_extraction import extract_features
import matplotlib.pyplot as plt

def show_results(query_img, top_indices, distances, image_paths):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Anh truy van")
    axs[0].axis("off")

    for i in range(3):
        img = cv2.imread(image_paths[top_indices[i]])
        axs[i + 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i + 1].set_title(f"Top {i+1}\nKhoang cach: {distances[i]:.2f}")
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

    print("Top 3 most similar")
    for i, idx in enumerate(top_k):
        print(f"Top {i+1}: {img_paths[idx]} - Khoang cach: {distances[idx]:.2f}")

    show_results(query_img, top_k, distances[top_k], img_paths)

def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if filepath:
        image = Image.open(filepath)
        image = image.resize((256, 256))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
        search(filepath)

app = tk.Tk()
app.title("tim anh giong nhat")
app.geometry("300x400")

btn = tk.Button(app, text="Chon anh", command=open_file)
btn.pack(pady=10)

image_label = tk.Label(app)
image_label.pack(pady=10)

app.mainloop()