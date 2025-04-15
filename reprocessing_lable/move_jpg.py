import os
import shutil

def move_jpg_files(source_dir, target_dir):
    """
    Di chuyển tất cả file .jpg từ thư mục nguồn (source_dir) sang thư mục đích (target_dir).
    """
    # Kiểm tra xem thư mục đích có tồn tại không, nếu không thì tạo mới
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Lặp qua tất cả các file trong thư mục nguồn
    for filename in os.listdir(source_dir):
        # Kiểm tra nếu file có đuôi .jpg
        if filename.endswith('.jpg'):
            # Đường dẫn đầy đủ đến file nguồn
            source_file = os.path.join(source_dir, filename)
            # Đường dẫn đầy đủ đến file đích
            target_file = os.path.join(target_dir, filename)
            # Di chuyển file
            shutil.move(source_file, target_file)
            print(f"Đã di chuyển: {filename}")

# Ví dụ sử dụng
if __name__ == "__main__":
    # Đường dẫn đến thư mục nguồn
    source_dir = r'dataset\val_pre\images'
    # Đường dẫn đến thư mục đích
    target_dir = r'dataset\images\val'

    # Di chuyển các file .jpg
    move_jpg_files(source_dir, target_dir)
    print("Hoàn tất di chuyển file!")