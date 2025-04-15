import cv2
import os
import torch
from ultralytics import YOLO  # Dùng cho YOLOv8
import numpy as np

# Lấy thư mục hiện tại của file Python
current_dir = os.path.dirname(__file__)

# Load mô hình YOLO đã huấn luyện
model_path = os.path.join(current_dir, "../../yolo11n.pt")  # Đường dẫn tương đối đến mô hình
model = YOLO(model_path)

# Đọc video từ đường dẫn
video_path = os.path.join(current_dir, "../../dataset/test_video.mp4/28291-369325225_tiny.mp4")  # Đường dẫn tương đối đến video
cam = cv2.VideoCapture(video_path)

# Tạo thư mục lưu kết quả nếu chưa có
output_folder = os.path.join(current_dir, "vid_results")  # Đường dẫn tương đối đến thư mục kết quả
os.makedirs(output_folder, exist_ok=True)

# Danh sách class cần giữ lại
CLASS_FILTER = {1, 2, 3, 4, 5, 6, 7, 8}

# Kiểm tra nếu video không mở được
if not cam.isOpened():
    print("❌ Không thể mở video, kiểm tra lại đường dẫn!")
    exit()

# Lấy thông tin video
frame_width = int(cam.get(3))
frame_height = int(cam.get(4))
frame_fps = int(cam.get(cv2.CAP_PROP_FPS))

# Tạo video writer để lưu kết quả
output_video_path = os.path.join(output_folder, "output_filtered.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, frame_fps, (frame_width, frame_height))

currentframe = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break  # Thoát nếu hết video

    # Chạy YOLO để dự đoán vật thể trên frame
    results = model(frame)

    for result in results:
        filtered_boxes = [box for box in result.boxes if int(box.cls[0].item()) in CLASS_FILTER]
        
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = f"{model.names[class_id]}: {confidence:.2f}"

            # Vẽ bounding box lên frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Ghi frame vào video kết quả
    out.write(frame)
    
    # Hiển thị frame
    cv2.imshow("Filtered YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Nhấn 'q' để thoát
    
    print(f"Processed frame {currentframe}")
    currentframe += 1

# Giải phóng bộ nhớ
cam.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Video đã lưu tại: {output_video_path}")
