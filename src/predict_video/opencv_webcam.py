import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os

# Lấy thư mục hiện tại của file Python
current_dir = os.path.dirname(__file__)

# Load mô hình YOLO đã huấn luyện (Thay "best.pt" bằng mô hình của bạn)
model_path = os.path.join(current_dir, "../../yolo11n.pt")  # Đường dẫn tương đối đến mô hình
model = YOLO(model_path)

# Khởi tạo webcam
cap = cv2.VideoCapture(0)  # Webcam mặc định, nếu có nhiều camera, thử 1, 2...

# Kiểm tra nếu webcam không mở được
if not cap.isOpened():
    print("Không thể mở webcam")
    exit()

while True:
    # Đọc frame từ webcam
    ret, frame = cap.read()
    if not ret:
        break  # Nếu không nhận được frame, thoát vòng lặp

    # Chạy YOLO để dự đoán vật thể trên frame
    results = model(frame)

    # Vẽ bounding box lên frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ bbox
            confidence = float(box.conf[0])  # Độ tin cậy
            class_id = int(box.cls[0])  # ID lớp dự đoán
            label = f"{model.names[class_id]}: {confidence:.2f}"  # Nhãn + xác suất

            # Vẽ khung chữ nhật (bounding box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị kết quả trực tiếp
    cv2.imshow("YOLO Detection", frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
