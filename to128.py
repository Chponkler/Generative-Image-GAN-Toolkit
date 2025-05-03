import os
from PIL import Image

# Параметры
input_folder = "/content/dataKup"     # Путь к исходной папке
output_folder = "/content/data128x128"  # Путь к папке для сохранения
target_size = 128

# Создаем выходную папку, если ее нет
os.makedirs(output_folder, exist_ok=True)

# Перебираем все изображения в папке
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path_in = os.path.join(input_folder, filename)
    path_out = os.path.join(output_folder, filename)

    try:
        img = Image.open(path_in).convert("RGB")
        w, h = img.size

        # Обрезка квадрата по центру
        min_side = min(w, h)
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        right = left + min_side
        bottom = top + min_side
        img_cropped = img.crop((left, top, right, bottom))

        # Масштабирование до 128x128
        img_resized = img_cropped.resize((target_size, target_size), Image.LANCZOS)

        # Сохраняем
        img_resized.save(path_out)
        print(f"Processed: {filename}")
    except Exception as e:
        print(f"Failed: {filename} ({e})")
