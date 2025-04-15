from ultralytics import YOLO
import os
import sys

# Lấy thư mục hiện tại của file Python
current_dir = os.path.dirname(__file__)

# Định nghĩa đường dẫn
model_path = os.path.join(current_dir, "../../yolo11n.pt")  # Đường dẫn tương đối đến model
image_dir = os.path.join(current_dir, "../../dataset/test")  # Đường dẫn tương đối đến thư mục ảnh test
output_dir = os.path.join(current_dir, "../../results/predict")  # Đường dẫn tương đối đến thư mục lưu kết quả

# Danh sách class cần nhận diện (ID từ 1 - 8)
CLASS_FILTER = {1, 2, 3, 4, 5, 6, 7, 8}

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

# Chạy dự đoán
print(f"🔍 Đang nhận diện {len(image_files)} ảnh...")
results = model(image_files)

# Đảm bảo thư mục output/predict tồn tại
os.makedirs(output_dir, exist_ok=True)

# Lọc kết quả chỉ giữ lại các class từ 1-8 và lưu vào thư mục "predict"
for image_path, result in zip(image_files, results):
    # Chỉ giữ lại bounding boxes có class từ 1-8
    filtered_boxes = [box for box in result.boxes if int(box.cls[0].item()) in CLASS_FILTER]
    
    if filtered_boxes:
        result.boxes = filtered_boxes  # Cập nhật lại boxes

        # Giữ nguyên tên file gốc
        file_name = os.path.basename(image_path)  
        save_path = os.path.join(output_dir, file_name)  

        # Lưu kết quả vào folder "predict"
        result.save(filename=save_path)  
        
        print(f"✅ Đã lưu kết quả: {save_path}")

print("✅ Hoàn thành! Kết quả được lưu trong", output_dir)
