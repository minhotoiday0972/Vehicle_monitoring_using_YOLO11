import os
import cv2
import torch
import numpy as np
import yt_dlp
from ultralytics import YOLO
import streamlit as st
import time
from collections import deque
import pandas as pd

# Lấy thư mục hiện tại của file Python
current_dir = os.path.dirname(__file__)

# Load YOLO model
model_path = os.path.join(current_dir, "yolo11n.pt")  # Sử dụng đường dẫn tương đối
model = YOLO(model_path)
if torch.cuda.is_available():
    model.to('cuda')
    print("Model chuyển sang CUDA thành công!")
else:
    print("CUDA không khả dụng, model vẫn chạy trên CPU.")

# Class filter và tên
CLASS_NAMES = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
CLASS_FILTER = set(CLASS_NAMES.keys())

# Streamlit UI
st.set_page_config(page_title="YOLOv11 Traffic Monitor", page_icon="🚦", layout="wide")
st.title("🚦 YOLOv11 Giám Sát Tình Trạng Giao Thông - Multi Stream")

# Sidebar
with st.sidebar:
    st.header("⚙️ Bảng trạng thái")
    st.markdown("Điều chỉnh các ngưỡng và thời gian để phù hợp với tình hình thực tế.")
    thong_thoang_threshold = st.number_input("Ngưỡng 'Thông thoáng' (boxes/5s):", min_value=0, value=5)
    binh_thuong_threshold = st.number_input("Ngưỡng 'Bình thường' (boxes/5s):", min_value=thong_thoang_threshold, value=10)
    ket_xe_threshold = st.number_input("Ngưỡng 'Kẹt xe' (boxes/5s):", min_value=binh_thuong_threshold, value=25)
    time_threshold_seconds = st.number_input("Thời gian duy trì trạng thái (giây):", min_value=1, value=10)

# Main area
st.markdown("Phương tiện phát hiện được, ứng dụng sẽ thông báo tình trạng giao thông (Thông thoáng, Bình thường, Đông đúc, Kẹt xe).")

col1, col2, col3 = st.columns(3)
status_boxes = [col1.empty(), col2.empty(), col3.empty()]  # Placeholders for status boxes

# Get stream URLs
def get_youtube_stream_url(url):
    ydl_opts = {'quiet': True, 'format': 'best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

video_urls = [
    "https://www.youtube.com/watch?v=7gVcaJMfpVI",
    "https://www.youtube.com/watch?v=muijHPW82vI",
    "https://www.youtube.com/watch?v=jysOgkivUKQ"
]
stream_urls = [get_youtube_stream_url(url) for url in video_urls]
caps = [cv2.VideoCapture(url) for url in stream_urls]

TARGET_SIZE = (640, 480)
box_counts = [deque(maxlen=5) for _ in caps]
status_history = [deque(maxlen=time_threshold_seconds) for _ in caps]
video_placeholder = st.empty()
status_table_placeholder = st.sidebar.empty()  # Placeholder for status table in sidebar

# Hàm xếp frame dạng lưới
def stack_frames(frames, rows, cols):
    h, w = TARGET_SIZE[1], TARGET_SIZE[0]
    blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
    grid_frames = [frames[i] if i < len(frames) else blank_frame for i in range(rows * cols)]
    row_frames = [np.hstack(grid_frames[i * cols:(i + 1) * cols]) for i in range(rows)]
    return np.vstack(row_frames)

# Chạy giám sát
def get_status(avg_boxes):
    if avg_boxes <= thong_thoang_threshold:
        return "Thông thoáng"
    elif avg_boxes <= binh_thuong_threshold:
        return "Bình thường"
    elif avg_boxes <= ket_xe_threshold:
        return "Đông đúc"
    else:
        return "Kẹt xe"

status_names_vn = {"Thông thoáng": "Thông thoáng", "Bình thường": "Bình thường", "Đông đúc": "Đông đúc", "Kẹt xe": "Kẹt xe"}
status_colors = {"Thông thoáng": "#a8f0c6", "Bình thường": "#cbf0a8", "Đông đúc": "#f0e6a8", "Kẹt xe": "#f0a8a8"}  # Define colors for each status

try:
    start_time = time.time()
    stream_status_list = []  # List to store stream status for table
    while True:
        frames = []
        stream_status_list = []  # Reset list in each loop
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                frames.append(np.zeros(TARGET_SIZE[::-1] + (3,), dtype=np.uint8))
                stream_status_list.append([f"Luồng {i+1}", "-", "-", "Mất kết nối"])  # Indicate lost connection in table
                continue

            resized_frame = cv2.resize(frame, TARGET_SIZE)
            results = model(resized_frame)[0]

            count = 0
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if class_id in CLASS_FILTER:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{CLASS_NAMES[class_id]} {confidence:.2f}"
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(resized_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            box_counts[i].append(count)
            avg_boxes = sum(box_counts[i]) / len(box_counts[i])
            status = get_status(avg_boxes)
            status_history[i].append(status)

            # Kiểm tra trạng thái duy trì
            current_status = "Đang cập nhật..."  # Default status
            if len(set(status_history[i])) == 1 and len(status_history[i]) == time_threshold_seconds:
                current_status = status
            else:
                current_status = get_status(avg_boxes)  # Show immediate status if not stable

            # Display status in box
            with status_boxes[i]:
                status_color = status_colors.get(current_status, "#ffffff")  # Default white if status not in dict
                st.markdown(f"<div style='background-color: {status_color}; padding: 10px; border-radius: 5px; text-align: center; color: black;'><b>Luồng {i + 1}: {status_names_vn.get(current_status, current_status)}</b></div>", unsafe_allow_html=True)

            # Update status in table
            stream_status_list.append([f"Luồng {i+1}", count, f"{avg_boxes:.2f}", current_status])

            frames.append(resized_frame)

        grid_size = int(np.ceil(np.sqrt(len(frames))))
        stacked_frame = stack_frames(frames, grid_size, grid_size)
        video_placeholder.image(stacked_frame, channels="BGR")

        # Display status table in sidebar
        status_df = pd.DataFrame(stream_status_list, columns=["Luồng", "1s Count", "5s Avg", "Tình trạng"])
        status_table_placeholder.dataframe(status_df, hide_index=True, use_container_width=True)

        # Giải phóng bộ nhớ GPU định kỳ
        if time.time() - start_time > 60:
            torch.cuda.empty_cache()
            start_time = time.time()
        time.sleep(0.1)  # Add a small delay to reduce CPU usage

except Exception as e:
    st.error(f"❌ Đã xảy ra lỗi: {e}")

finally:
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    st.success("✅ Kết thúc chương trình!")