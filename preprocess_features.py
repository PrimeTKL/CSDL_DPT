import os
from pathlib import Path
# Import hàm trích xuất đặc trưng
from BTLHCSDLDPT.SIFT_Histogram import extract_sift_features # <--- Đảm bảo import hàm này

# --- CẤU HÌNH ---
# THAY THẾ DÒNG NÀY BẰNG ĐƯỜNG DẪN THỰC TẾ ĐẾN THƯ MỤC CẦN TRÍCH XUẤT ĐẶC TRƯNG (ví dụ: thư mục dataset)
images_folder_to_extract = "D:/CSDLDPT/BTLHCSDLDPT/dataset_images" # Hoặc dùng os.path.join

# Thư mục lưu cache đặc trưng
cache_dir = "features_cache" # Nên giống với thư mục cache dùng trong script tìm kiếm

# Các định dạng ảnh được hỗ trợ
supported_image_extensions = ('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')


# --- THU THẬP DANH SÁCH ẢNH ---
print(f"Đang thu thập danh sách ảnh từ thư mục: {images_folder_to_extract}")
image_files = []
if os.path.isdir(images_folder_to_extract):
    for entry in os.listdir(images_folder_to_extract):
        full_path = os.path.join(images_folder_to_extract, entry)
        if os.path.isfile(full_path) and entry.lower().endswith(supported_image_extensions):
            image_files.append(full_path)
    print(f"Tìm thấy tổng cộng {len(image_files)} ảnh cần trích xuất đặc trưng.")
else:
    print(f"Lỗi: Thư mục không tồn tại hoặc không phải là thư mục: {images_folder_to_extract}")


# --- TRÍCH XUẤT ĐẶC TRƯNG CHO TỪNG ẢNH ---
if image_files:
    print("\n--- Bắt đầu trích xuất đặc trưng ---")
    for img_path in image_files:
        # Gọi hàm trích xuất đặc trưng. Hàm này sẽ tự động dùng/tạo cache.
        keypoints, descriptors = extract_sift_features(img_path, features_cache=cache_dir)
        # Bạn có thể thêm print ở đây để biết ảnh nào đang được xử lý
        # print(f"Đã xử lý ảnh: {img_path}")

    print("\n--- Hoàn thành quá trình trích xuất đặc trưng ---")
    print(f"Đặc trưng đã được lưu (hoặc đã tồn tại) trong thư mục cache: {cache_dir}")
else:
    print("Không có ảnh nào để trích xuất đặc trưng.")