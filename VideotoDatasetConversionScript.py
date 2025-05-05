#код для преобразования видео в скриншоты
import cv2
import os

# ==== Параметры ====
video_path = "/content/drive/MyDrive/dataset/videoplayback.mp4"             # Укажи путь к видео
output_dir = "video_screenshots"          # Папка для сохранённых изображений
interval = 4                              # Интервал (секунд)
crop_x, crop_y = 0, 250                     # Левый верхний угол
crop_w, crop_h = 250, 205                 # Размер обрезки

# ==== Подготовка ====
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Не удалось открыть видео: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    raise ValueError("FPS видео равно 0 — возможно, видео повреждено или неподдерживаемый формат.")

frame_interval = int(fps * interval)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_num = 0
saved_idx = 0

while frame_num < total_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    if crop_x + crop_w <= w and crop_y + crop_h <= h:
        crop = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        out_path = os.path.join(output_dir, f"crop1_{saved_idx:04d}.jpg")
        cv2.imwrite(out_path, crop)
        print(f"Saved: {out_path}")
        saved_idx += 1
    else:
        print(f"Frame {frame_num}: Crop out of bounds, skipping.")

    frame_num += frame_interval

cap.release()
print("✅ Готово.")
