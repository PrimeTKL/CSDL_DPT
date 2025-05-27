import numpy as np

# Đọc danh sách vector đặc trưng
features = np.load("features.npy", allow_pickle=True)

# Đọc danh sách đường dẫn ảnh tương ứng
image_paths = np.load("image_paths.npy", allow_pickle=True)

# Kiểm tra số lượng
print(features[0][*]) 
print("Số lượng ảnh:", len(image_paths))

print("vector đặc trưng của ảnh đầu tiên:", features.shape)
