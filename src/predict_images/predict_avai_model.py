from ultralytics import YOLO
import os
import sys

# Lấy thư mục hiện tại của file Python
current_dir = os.path.dirname(__file__)

# Định nghĩa đường dẫn
model_path = os.path.join(current_dir, "../../yolo11n.pt")  # Đường dẫn tương đối đến model
image_dir = os.path.join(current_dir, "../../dataset/test")  # Đường dẫn tương đối đến thư mục ảnh test
output_dir = os.path.join(current_dir, "../../results")  # Đường dẫn tương đối đến thư mục lưu kết quả

# Kiểm tra sự tồn tại của model đã train
if not os.path.exists(model_path):
    print("❌ Lỗi: Không tìm thấy tệp yolo11n.pt. Hãy đảm bảo bạn đã huấn luyện xong và model được lưu đúng cách!")
    sys.exit()

# Load mô hình đã được train
try:
    model = YOLO(model_path)
    print("✅ Model YOLO đã được load thành công!")
except Exception as e:
    print(f"❌ Lỗi khi load model: {e}")
    sys.exit()

# Kiểm tra thư mục chứa ảnh test
if not os.path.exists(image_dir):
    print(f"❌ Lỗi: Không tìm thấy thư mục {image_dir}")
    sys.exit()

# Lấy danh sách tất cả ảnh hợp lệ trong thư mục
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

if not image_files:
    print(f"❌ Lỗi: Không tìm thấy ảnh hợp lệ trong {image_dir}")
    sys.exit()

# Dự đoán trên tất cả ảnh và lưu kết quả
print(f"🔍 Đang chạy dự đoán trên {len(image_files)} ảnh...")
results = model(image_files, save=True, project=output_dir)

print(f"✅ Dự đoán hoàn tất! Kết quả được lưu trong {output_dir}")
