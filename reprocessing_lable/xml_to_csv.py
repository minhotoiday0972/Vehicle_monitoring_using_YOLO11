import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(os.path.join(path, "*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Lấy tên file ảnh từ XML (nếu không có <path> thì lấy từ tên file XML)
        filename = root.find("path")
        if filename is not None:
            filename = os.path.basename(filename.text)
        else:
            filename = os.path.basename(xml_file).replace(".xml", ".jpg")  # Giả định ảnh là .jpg
        
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        for member in root.findall("object"):
            class_name = member.find("name").text
            xmin = int(member.find("bndbox/xmin").text)
            ymin = int(member.find("bndbox/ymin").text)
            xmax = int(member.find("bndbox/xmax").text)
            ymax = int(member.find("bndbox/ymax").text)

            xml_list.append((filename, width, height, class_name, xmin, ymin, xmax, ymax))

    # Tạo DataFrame
    column_name = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    
    return xml_df


def main():
    # Định nghĩa đường dẫn thư mục chứa file XML
    directories = [r"dataset\train\images"]

    for directory in directories:
        # Đảm bảo đường dẫn hợp lệ
        image_path = os.path.abspath(directory)
        
        # Chuyển đổi XML -> CSV
        xml_df = xml_to_csv(image_path)

        # Tạo tên file CSV tương ứng
        output_csv = os.path.join(r"dataset\train\labels", "train_labels.csv")

        # Lưu CSV
        xml_df.to_csv(output_csv, index=False)
        
        print(f"Successfully converted XML to CSV: {output_csv}")


if __name__ == "__main__":
    main()
