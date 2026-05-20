"""
Определяет категорию каждой вещи в preprocessed/ с помощью CLIP zero-shot classification
и раскладывает PNG по папкам (tops/, pants/, shoes/, outerwear/, accessories/).

Zero-shot — значит без обучения: CLIP сравнивает изображение с текстовыми описаниями
категорий и выбирает наиболее похожее. Это работает потому что CLIP обучен на
миллионах пар изображение-текст и понимает визуальный смысл слов.
"""

import os
import shutil
from pathlib import Path

import certifi
import torch

# На macOS Python часто не находит системные SSL-сертификаты.
# Указываем вручную на сертификаты из пакета certifi — это стандартное решение.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


PREPROCESSED_DIR = Path(__file__).parent.parent / "preprocessed"

# Текстовые описания категорий — чем точнее формулировка, тем лучше результат.
# CLIP сравнит изображение с каждым из этих текстов и выберет самый похожий.
CATEGORIES = {
    "tops":        "a photo of a top, t-shirt, shirt, blouse, or sweater",
    "pants":       "a photo of pants, jeans, trousers, or shorts",
    "skirts":      "a photo of a skirt, mini skirt, midi skirt, or maxi skirt",
    "shoes":       "a photo of shoes, sneakers, boots, or sandals",
    "outerwear":   "a photo of a jacket, coat, hoodie, or outerwear",
    "accessories": "a photo of an accessory, bag, hat, scarf, or belt",
}


def load_clip():
    """Загружает CLIP модель и процессор. При первом запуске скачивает ~600MB."""
    print("Загружаю CLIP модель...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()  # режим инференса, не обучения
    return model, processor


def classify_image(image_path: Path, model, processor) -> str:
    """
    Определяет категорию одного изображения.
    Возвращает ключ из словаря CATEGORIES (например, "tops").
    """
    image = Image.open(image_path).convert("RGBA")

    # Белый фон нужен потому что CLIP обучался на обычных фото без прозрачности.
    # Прозрачные пиксели превратились бы в чёрные и сбили бы модель.
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.paste(image, mask=image)
    image_rgb = background.convert("RGB")

    # processor готовит и изображение и тексты в формат который понимает CLIP
    inputs = processor(
        text=list(CATEGORIES.values()),
        images=image_rgb,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():  # не считаем градиенты — нам не нужно обучение
        outputs = model(**inputs)

    # logits_per_image — матрица схожести изображения с каждым текстом
    # softmax превращает числа в вероятности (сумма = 1)
    probs = outputs.logits_per_image.softmax(dim=1)[0]

    # выбираем категорию с наибольшей вероятностью
    best_idx = probs.argmax().item()
    category_name = list(CATEGORIES.keys())[best_idx]
    confidence = probs[best_idx].item()

    return category_name, confidence


def classify_all():
    # берём только PNG прямо в preprocessed/, не заходя в подпапки категорий
    png_files = [
        f for f in PREPROCESSED_DIR.iterdir()
        if f.is_file() and f.suffix.lower() == ".png"
    ]

    if not png_files:
        print(f"Нет PNG файлов в {PREPROCESSED_DIR}.")
        print("Сначала запусти scripts/preprocess_images.py")
        return

    model, processor = load_clip()

    print(f"\nКлассифицирую {len(png_files)} изображений...\n")

    for image_path in png_files:
        category, confidence = classify_image(image_path, model, processor)

        # перемещаем файл в папку категории
        dest = PREPROCESSED_DIR / category / image_path.name
        shutil.move(str(image_path), str(dest))

        print(f"  {image_path.name}")
        print(f"  -> {category}/ (уверенность: {confidence:.0%})\n")

    print("Готово!")


if __name__ == "__main__":
    classify_all()
