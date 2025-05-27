import numpy as np
from sklearn.model_selection import train_test_split

X = np.load("bird_features.npy", allow_pickle=True)
y = np.load("bird_labels.npy", allow_pickle=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def compute_topk_accuracy(X_train, y_train, X_test, y_test, k=3):
    correct_top1 = 0
    correct_topk = 0

    for i, x_query in enumerate(X_test):
        distances = np.linalg.norm(X_train - x_query, axis=1)
        top_k = np.argsort(distances)[:k]

        if y_test[i] == y_train[top_k[0]]:
            correct_top1 += 1
        if y_test[i] in y_train[top_k]:
            correct_topk += 1

    top1_acc = correct_top1 / len(X_test)
    topk_acc = correct_topk / len(X_test)

    return top1_acc, topk_acc

top1, top3 = compute_topk_accuracy(X_train, y_train, X_test, y_test, k=3)
print(f"Top-1 Accuracy: {top1 * 100:.2f}%")
print(f"Top-3 Accuracy: {top3 * 100:.2f}%")