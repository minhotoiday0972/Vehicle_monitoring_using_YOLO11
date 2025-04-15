import subprocess
import cv2

def get_earthcam_stream(url):
    try:
        result = subprocess.run(
            ["yt-dlp", "-g", url],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"❌ Lỗi khi lấy stream: {e}")
        return None

camera_url = get_earthcam_stream("https://videos-3.earthcam.com/fecnetwork/4280.flv/chunklist_w1712197052.m3u8")

if camera_url:
    print(f"🎥 Streaming từ: {camera_url}")
    cap = cv2.VideoCapture(camera_url)
else:
    print("❌ Không tìm thấy luồng video hợp lệ.")
