import cv2
import os
import torch
from ultralytics import YOLO
import numpy as np

# Lấy thư mục hiện tại của file Python
current_dir = os.path.dirname(__file__)

# Load mô hình YOLO đã huấn luyện (Thay đổi model theo phiên bản bạn dùng)
model_path = os.path.join(current_dir, "../../yolo11n.pt")  # Đường dẫn tương đối đến mô hình
model = YOLO(model_path)

# Đọc video từ đường dẫn
video_path = os.path.join(current_dir, "../../dataset/test_video.mp4/28291-369325225_tiny.mp4")  # Đường dẫn tương đối đến video
cam = cv2.VideoCapture(video_path)

# Tạo thư mục lưu ảnh nếu chưa có
output_folder = os.path.join(current_dir, "vid_results")  # Đường dẫn tương đối đến thư mục kết quả
os.makedirs(output_folder, exist_ok=True)

# Biến đếm số frame
currentframe = 0

while True:
    # Đọc frame từ video
    ret, frame = cam.read()
    if not ret:
        break  # Thoát vòng lặp nếu hết video

    # Chạy YOLO để dự đoán vật thể trên frame
    results = model(frame)

    # Vẽ bounding box lên frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ bbox
            confidence = float(box.conf[0])  # Xác suất
            class_id = int(box.cls[0])  # ID lớp dự đoán
            label = f"{model.names[class_id]}: {confidence:.2f}"  # Tên lớp và xác suất

            # Vẽ khung chữ nhật (bounding box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Lưu frame đã xử lý
    output_path = os.path.join(output_folder, f"frame{currentframe}.jpg")
    cv2.imwrite(output_path, frame)

    print(f"Processed {output_path}")

    # Tăng biến đếm frame
    currentframe += 1

# Giải phóng bộ nhớ
cam.release()
cv2.destroyAllWindows()
