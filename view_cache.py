import pickle
import os
import cv2 # Cần thiết để hiểu đối tượng cv2.KeyPoint sau khi deserialize
from pathlib import Path

# --- Hàm trợ giúp giải sé-ri hóa KeyPoint (Copy từ SIFT_Histogram.py) ---
# Hàm này cần có để chuyển dữ liệu KeyPoint từ định dạng lưu trữ về đối tượng cv2.KeyPoint
# Đảm bảo hàm này khớp với phiên bản bạn đã dùng để LƯU (serialize).
def deserialize_keypoints(serialized_keypoints):
    """Chuyển định dạng pickleable ngược lại thành danh sách cv2.KeyPoint."""
    if not isinstance(serialized_keypoints, list):
        print("Cảnh báo: Dữ liệu keypoint tải lên không phải là danh sách.")
        return None # Hoặc xử lý lỗi tùy ý

    # Kiểm tra số thuộc tính cho mỗi keypoint trong dữ liệu tải lên
    # Hàm pickle lưu theo tuple 7 thuộc tính
    # Kiểm tra nếu dữ liệu là danh sách các tuple và mỗi tuple có 7 phần tử
    if all(isinstance(kp_data, tuple) and len(kp_data) == 7 for kp_data in serialized_keypoints):
        return [cv2.KeyPoint(x, y, _size, _angle, _response, _octave, _class_id)
                for x, y, _size, _angle, _response, _octave, _class_id in serialized_keypoints]
    else:
         print("Cảnh báo: Dữ liệu keypoint có thể không đúng định dạng tuple 7 thuộc tính.")
         # Nếu không đúng định dạng 7 thuộc tính, thử tạo KeyPoint với số thuộc tính có sẵn
         # Điều này có thể gây lỗi nếu thiếu các thuộc tính bắt buộc (ví dụ: size)
         try:
             return [cv2.KeyPoint(x, y, *rest) # Sử dụng *rest để truyền các thuộc tính còn lại
                     for x, y, *rest in serialized_keypoints if isinstance(rest, tuple)]
         except Exception as e:
             print(f"Lỗi khi cố gắng giải sé-ri hóa KeyPoint với định dạng không chuẩn: {e}")
             return None # Trả về None nếu không giải sé-ri hóa được


# --- CẤU HÌNH ---
# Thay thế bằng đường dẫn THỰC TẾ đến thư mục cache của bạn
cache_folder_path = "features_cache" # Ví dụ: "D:/CSDLDPT/features_cache" nếu bạn chạy script từ D:\CSDLDPT

# --- THU THẬP DANH SÁCH CÁC FILE .pkl TRONG THƯ MỤC CACHE ---
print(f"Đang tìm kiếm các file .pkl trong thư mục cache: {cache_folder_path}")
pkl_files = []
if os.path.isdir(cache_folder_path):
    for entry in os.listdir(cache_folder_path):
        full_path = os.path.join(cache_folder_path, entry)
        if os.path.isfile(full_path) and entry.lower().endswith('.pkl'):
            pkl_files.append(full_path)
    print(f"Tìm thấy tổng cộng {len(pkl_files)} file .pkl.")
else:
    print(f"Lỗi: Thư mục cache không tồn tại hoặc không phải là thư mục: {cache_folder_path}")


# --- ĐỌC VÀ HIỂN THỊ NỘI DUNG CỦA TỪNG FILE .pkl ---
if pkl_files:
    print("\n===== HIỂN THỊ NỘI DUNG CÁC FILE CACHE =====")
    for pkl_file_path in pkl_files:
        print(f"\n--- Đọc file: {pkl_file_path} ---")

        try:
            with open(pkl_file_path, 'rb') as f:
                # Tải dữ liệu từ file .pkl
                data = pickle.load(f)

                # Dữ liệu được lưu dưới dạng tuple (serialized_keypoints, descriptors)
                if isinstance(data, tuple) and len(data) == 2:
                    serialized_keypoints, descriptors = data

                    # Chuyển đổi picklable_keypoints về lại định dạng gốc của OpenCV
                    keypoints = deserialize_keypoints(serialized_keypoints)


                    print("  --- Thông tin Keypoints (Điểm đặc trưng) ---")
                    if keypoints is not None:
                         print(f"  Số lượng keypoints: {len(keypoints)}")
                         if keypoints:
                             # In thông tin của vài keypoint đầu tiên
                             print("  Một vài keypoint đầu tiên (tọa độ x, y, size, angle, response, octave, class_id):")
                             # Chỉ in tối đa 5 keypoint hoặc ít hơn nếu danh sách có ít hơn 5
                             for i, kp in enumerate(keypoints[:min(len(keypoints), 5)]):
                                print(f"    Keypoint {i+1}: ({kp.pt[0]:.2f}, {kp.pt[1]:.2f}, size={kp.size:.2f}, angle={kp.angle:.2f}, response={kp.response:.2f}, octave={kp.octave}, class_id={kp.class_id})")
                         else:
                              print("  Không tìm thấy keypoints nào trong file này.")
                    else:
                         print("  Không thể giải sé-ri hóa Keypoints.")


                    print("  --- Thông tin Descriptors (Vector đặc trưng) ---")
                    if descriptors is not None:
                        print(f"  Hình dạng (shape) của descriptors: {descriptors.shape}") # Shape là (số lượng keypoint, 128)
                        if descriptors.shape[0] > 0:
                             print("  Một vài descriptor đầu tiên (chỉ hiển thị giá trị của descriptor đầu tiên):")
                             # In ra 128 giá trị của descriptor đầu tiên
                             print("  ", descriptors[0])
                        else:
                             print("  Không có descriptors nào trong file này.")

                    else:
                        print("  Không có descriptors được lưu.")

                else:
                    print("  Lỗi: Dữ liệu trong file .pkl không đúng định dạng (không phải tuple 2 phần).")

        except Exception as e:
            print(f"  Lỗi khi đọc file pickle: {e}")

    print("\n===== Hoàn thành hiển thị nội dung cache =====")
else:
    print("Không tìm thấy file .pkl nào để hiển thị.")