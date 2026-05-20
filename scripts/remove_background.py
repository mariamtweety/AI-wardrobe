"""
Скрипт удаляет фон у всех изображений из папки raw/
и сохраняет результат в processed/ в формате PNG с прозрачностью.
"""

from pathlib import Path
from PIL import Image
from rembg import remove


# Папки относительно корня проекта (wardrobe-ai/)
RAW_DIR = Path(__file__).parent.parent / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "processed"

# Форматы, которые умеем читать
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def remove_background(image_path: Path) -> Image.Image:
    """
    Читает изображение и удаляет фон через rembg.
    Возвращает PIL-изображение с прозрачным фоном (режим RGBA).
    """
    with open(image_path, "rb") as f:
        input_data = f.read()  # читаем файл как байты

    output_data = remove(input_data)  # rembg возвращает PNG-байты с прозрачностью

    # Превращаем байты обратно в PIL-изображение, чтобы удобно сохранять
    from io import BytesIO
    return Image.open(BytesIO(output_data)).convert("RGBA")


def process_all_images():
    """
    Проходит по всем изображениям в raw/, удаляет фон, сохраняет в processed/.
    """
    # Собираем список файлов с нужными расширениями
    image_files = [
        f for f in RAW_DIR.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"Нет изображений в {RAW_DIR}. Добавь файлы и запусти снова.")
        return

    print(f"Найдено изображений: {len(image_files)}")

    for image_path in image_files:
        print(f"  Обрабатываю: {image_path.name} ...", end=" ")

        result = remove_background(image_path)

        # Имя выходного файла всегда .png — только PNG поддерживает прозрачность
        output_path = PROCESSED_DIR / (image_path.stem + ".png")
        result.save(output_path)

        print(f"-> сохранено: {output_path.name}")

    print("\nГотово!")


if __name__ == "__main__":
    process_all_images()
