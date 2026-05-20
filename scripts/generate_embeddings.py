"""
Генерирует CLIP-эмбеддинги для всех вещей в preprocessed/ и сохраняет в data/embeddings.npz.

Что такое эмбеддинг:
  Это список из 512 чисел, который описывает "смысл" изображения.
  Похожие вещи → похожие числа → близкие векторы в пространстве.
  Именно это позволит нам искать похожую одежду и генерировать outfit.

Почему 512:
  Такой размер у модели clip-vit-base-patch32. Это баланс между
  точностью и скоростью — достаточно для нашей задачи.
"""

import os
from pathlib import Path

import certifi
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


PREPROCESSED_DIR = Path(__file__).parent.parent / "preprocessed"
DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npz"

CATEGORIES = ["tops", "pants", "shoes", "outerwear", "accessories"]


def load_clip():
    print("Загружаю CLIP модель...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


def get_embedding(image_path: Path, model, processor) -> np.ndarray:
    """
    Превращает одно изображение в вектор (эмбеддинг).

    Нормализация (деление на длину вектора) важна для cosine similarity:
    мы хотим сравнивать направление векторов, а не их длину.
    """
    image = Image.open(image_path).convert("RGBA")

    # CLIP обучался на RGB — заменяем прозрачность белым фоном
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.paste(image, mask=image)
    image_rgb = background.convert("RGB")

    inputs = processor(images=image_rgb, return_tensors="pt")

    with torch.no_grad():
        # transformers 5.x возвращает объект — берём тензор через vision_model
        vision_output = model.vision_model(**inputs)
        embedding = model.visual_projection(vision_output.pooler_output)

    # normalize: превращаем вектор в единичный (длина = 1)
    # это стандарт для cosine similarity
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding[0].numpy()


def collect_images() -> list[tuple[Path, str]]:
    """Собирает все PNG из папок категорий. Возвращает список (путь, категория)."""
    items = []
    for category in CATEGORIES:
        category_dir = PREPROCESSED_DIR / category
        if not category_dir.exists():
            continue
        for png in category_dir.glob("*.png"):
            items.append((png, category))
    return items


def generate_embeddings():
    items = collect_images()

    if not items:
        print("Нет изображений в preprocessed/.")
        print("Сначала запусти classify_category.py")
        return

    model, processor = load_clip()

    print(f"\nГенерирую эмбеддинги для {len(items)} вещей...\n")

    filenames = []
    categories = []
    embeddings = []

    for image_path, category in items:
        print(f"  {image_path.name} [{category}] ...", end=" ")

        embedding = get_embedding(image_path, model, processor)
        embeddings.append(embedding)
        filenames.append(image_path.name)
        categories.append(category)

        print(f"готово, вектор: {embedding.shape}")

    # сохраняем всё в один файл .npz — это numpy-формат для нескольких массивов
    np.savez(
        EMBEDDINGS_FILE,
        filenames=np.array(filenames),
        categories=np.array(categories),
        embeddings=np.array(embeddings),
    )

    print(f"\nСохранено -> {EMBEDDINGS_FILE}")
    print(f"Форма матрицы: {np.array(embeddings).shape}")
    print("(строки = вещи, столбцы = 512 чисел эмбеддинга)")


if __name__ == "__main__":
    generate_embeddings()
