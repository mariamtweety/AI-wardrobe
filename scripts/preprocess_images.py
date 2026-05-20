"""
Preprocessing pipeline для изображений одежды.

Берёт PNG с прозрачным фоном из processed/,
и делает: crop -> resize -> padding -> сохраняет в preprocessed/.

Почему это важно для ML/CLIP:
  CLIP ожидает изображения фиксированного размера 224x224.
  Если просто растянуть — одежда деформируется и эмбеддинги будут хуже.
  Правильный путь: обрезать пустое пространство, сохранить пропорции,
  добавить прозрачные поля чтобы получить квадрат.
"""

from pathlib import Path
from PIL import Image


PROCESSED_DIR = Path(__file__).parent.parent / "processed"
PREPROCESSED_DIR = Path(__file__).parent.parent / "preprocessed"

TARGET_SIZE = 224  # стандартный размер для CLIP и большинства vision-моделей


def crop_to_content(image: Image.Image) -> Image.Image:
    """
    Обрезает прозрачные края вокруг одежды.

    Зачем: после удаления фона вокруг объекта остаётся
    много пустого прозрачного пространства. Без crop модель
    будет "видеть" маленький объект посреди пустоты — это
    ухудшает качество эмбеддингов.
    """
    # getbbox() находит минимальный прямоугольник вокруг непрозрачных пикселей
    bbox = image.getbbox()

    if bbox is None:
        # изображение полностью прозрачное — возвращаем как есть
        return image

    return image.crop(bbox)


def resize_with_padding(image: Image.Image, target_size: int) -> Image.Image:
    """
    Масштабирует изображение до target_size x target_size
    без деформации, добавляя прозрачные поля по краям.

    Зачем resize: модели принимают фиксированный размер входа.

    Зачем padding: одежда редко бывает квадратной (футболка шире чем длиннее,
    брюки длиннее чем шире). Если просто растянуть до квадрата — пропорции
    нарушатся. Padding заполняет недостающее пространство прозрачностью,
    объект остаётся нетронутым и отцентрированным.
    """
    # thumbnail() уменьшает изображение вписывая его в target_size x target_size,
    # сохраняя пропорции. Оригинал не изменяет — нужна копия.
    resized = image.copy()
    resized.thumbnail((target_size, target_size), Image.LANCZOS)

    # создаём чистый квадрат target_size x target_size с прозрачным фоном
    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    # считаем отступы чтобы разместить изображение по центру
    x_offset = (target_size - resized.width) // 2
    y_offset = (target_size - resized.height) // 2

    # вставляем изображение на холст; маска нужна чтобы прозрачность работала корректно
    canvas.paste(resized, (x_offset, y_offset), mask=resized)

    return canvas


def preprocess(image_path: Path) -> Image.Image:
    """
    Полный pipeline для одного изображения: crop -> resize -> padding.
    """
    image = Image.open(image_path).convert("RGBA")

    image = crop_to_content(image)
    image = resize_with_padding(image, TARGET_SIZE)

    return image


def process_all_images():
    png_files = list(PROCESSED_DIR.glob("*.png"))

    if not png_files:
        print(f"Нет PNG файлов в {PROCESSED_DIR}.")
        print("Сначала запусти scripts/remove_background.py")
        return

    print(f"Найдено изображений: {len(png_files)}")

    for image_path in png_files:
        print(f"  Обрабатываю: {image_path.name} ...", end=" ")

        result = preprocess(image_path)

        output_path = PREPROCESSED_DIR / image_path.name
        result.save(output_path)

        print(f"-> {result.width}x{result.height}  сохранено: {output_path.name}")

    print("\nГотово!")


if __name__ == "__main__":
    process_all_images()
