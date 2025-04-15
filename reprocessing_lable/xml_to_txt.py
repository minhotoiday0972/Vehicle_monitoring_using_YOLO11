import os
import xml.etree.ElementTree as ET

# Hàm chuyển đổi tọa độ từ PASCAL VOC sang YOLO
def convert_to_yolo_format(size, box):
    """
    Chuyển đổi tọa độ từ PASCAL VOC (xmin, ymin, xmax, ymax) sang YOLO (x_center, y_center, width, height).
    """
    dw = 1.0 / size[0]  # Tỷ lệ chiều rộng
    dh = 1.0 / size[1]  # Tỷ lệ chiều cao

    # Tính tọa độ tâm và kích thước bounding box
    x_center = (box[0] + box[2]) / 2.0 * dw
    y_center = (box[1] + box[3]) / 2.0 * dh
    width = (box[2] - box[0]) * dw
    height = (box[3] - box[1]) * dh

    return (x_center, y_center, width, height)

# Hàm xử lý file .xml và ghi vào file .txt
def convert_xml_to_txt(xml_file, output_dir, class_mapping):
    """
    Chuyển đổi file .xml sang file .txt theo định dạng YOLO.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Lấy kích thước ảnh
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # Tạo file .txt tương ứng
    txt_filename = os.path.splitext(os.path.basename(xml_file))[0] + '.txt'
    txt_filepath = os.path.join(output_dir, txt_filename)

    with open(txt_filepath, 'w') as f:
        for obj in root.findall('object'):
            # Lấy tên lớp và ánh xạ sang class_id
            class_name = obj.find('name').text
            if class_name not in class_mapping:
                continue  # Bỏ qua nếu lớp không có trong class_mapping
            class_id = class_mapping[class_name]

            # Lấy tọa độ bounding box
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Chuyển đổi tọa độ sang YOLO format
            yolo_box = convert_to_yolo_format((width, height), (xmin, ymin, xmax, ymax))

            # Ghi vào file .txt
            f.write(f"{class_id} {' '.join(map(str, yolo_box))}\n")

# Hàm chính để xử lý toàn bộ thư mục
def convert_all_xml_to_txt(xml_dir, output_dir, class_mapping):
    """
    Chuyển đổi tất cả file .xml trong thư mục sang file .txt.
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Lặp qua tất cả file .xml trong thư mục
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            convert_xml_to_txt(xml_path, output_dir, class_mapping)

# Ví dụ sử dụng
if __name__ == "__main__":
    # Lấy thư mục hiện tại của file Python
    current_dir = os.path.dirname(__file__)

    # Đường dẫn tương đối đến thư mục chứa file .xml
    xml_dir = os.path.join(current_dir, "../dataset/val_pre/images")
    # Đường dẫn tương đối đến thư mục đầu ra chứa file .txt
    output_dir = os.path.join(current_dir, "../dataset/labels/val")
    # Ánh xạ tên lớp sang class_id
    class_mapping = {'car': 0, 'motorcycle': 1, 'bus': 2, 'truck': 3}

    # Chuyển đổi tất cả file .xml sang .txt
    convert_all_xml_to_txt(xml_dir, output_dir, class_mapping)
    print("Chuyển đổi hoàn tất!")
