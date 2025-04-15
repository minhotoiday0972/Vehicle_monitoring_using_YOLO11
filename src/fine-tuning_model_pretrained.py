from ultralytics import YOLO
import os
import sys

# Lấy thư mục hiện tại của file Python
current_dir = os.path.dirname(__file__)

# Kiểm tra sự tồn tại của các file
model_path = os.path.join(current_dir, "../yolo11n.pt")  # Đường dẫn tương đối đến tệp mô hình
data_yaml_path = os.path.join(current_dir, "../dataset/data.yaml")  # Đường dẫn tương đối đến tệp data.yaml

if not os.path.exists(model_path):
    print("Lỗi: Không tìm thấy tệp yolo11n.pt")
    sys.exit()

if not os.path.exists(data_yaml_path):
    print("Lỗi: Không tìm thấy tệp data.yaml")
    sys.exit()

# Load model pretrained
model = YOLO(model_path)

# Huấn luyện mô hình
results = model.train(data=data_yaml_path, epochs=10, batch=16)
print(results)

# Đánh giá mô hình
results = model.val()
print(results)

# Xuất mô hình sang ONNX
try:
    success = model.export(format="onnx")
    if success:
        print("Xuất mô hình sang ONNX thành công!")
    else:
        print("Xuất mô hình thất bại.")
except Exception as e:
    print(f"Lỗi khi xuất ONNX: {e}")
