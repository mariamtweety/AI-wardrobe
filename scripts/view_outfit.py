"""
Показывает outfit визуально — склеивает вещи в одну картинку и открывает её.
Запуск: python scripts/view_outfit.py         # показывает последний outfit
        python scripts/view_outfit.py outfit_002  # показывает конкретный
"""

import json
import sys
from pathlib import Path

from PIL import Image


DATA_DIR = Path(__file__).parent.parent / "data"
PREPROCESSED_DIR = Path(__file__).parent.parent / "preprocessed"
OUTFITS_FILE = DATA_DIR / "outfits.json"

CATEGORIES = ["tops", "pants", "skirts", "shoes", "outerwear", "accessories"]
ITEM_SIZE = 300  # размер каждой вещи в коллаже


def find_image(filename: str) -> Path | None:
    """Ищет PNG по всем папкам категорий."""
    for category in CATEGORIES:
        path = PREPROCESSED_DIR / category / filename
        if path.exists():
            return path
    return None


def make_collage(outfit: dict) -> Image.Image:
    """Склеивает вещи из outfit в одну горизонтальную картинку."""
    images = []
    labels = []

    for filename, category in zip(outfit["items"], outfit["categories"]):
        path = find_image(filename)
        if path is None:
            print(f"  Файл не найден: {filename}")
            continue

        img = Image.open(path).convert("RGBA")

        # белый фон чтобы было видно на светлом фоне
        canvas = Image.new("RGBA", (ITEM_SIZE, ITEM_SIZE), (245, 245, 245, 255))
        img.thumbnail((ITEM_SIZE, ITEM_SIZE))
        x = (ITEM_SIZE - img.width) // 2
        y = (ITEM_SIZE - img.height) // 2
        canvas.paste(img, (x, y), mask=img)

        images.append(canvas)
        labels.append(category)

    if not images:
        raise ValueError("Ни одно изображение не найдено")

    # склеиваем горизонтально с небольшими отступами
    gap = 20
    total_width = ITEM_SIZE * len(images) + gap * (len(images) - 1)
    collage = Image.new("RGBA", (total_width, ITEM_SIZE), (255, 255, 255, 255))

    for i, img in enumerate(images):
        collage.paste(img, (i * (ITEM_SIZE + gap), 0))

    return collage.convert("RGB")


def view(outfit_id: str | None = None):
    with open(OUTFITS_FILE, "r", encoding="utf-8") as f:
        outfits = json.load(f)

    if not outfits:
        print("Нет сгенерированных outfit. Запусти generate_outfit.py")
        return

    if outfit_id == "all":
        for outfit in outfits:
            print(f"Показываю: {outfit['id']}  score: {outfit['score']}")
            make_collage(outfit).show()
        return

    if outfit_id:
        outfit = next((o for o in outfits if o["id"] == outfit_id), None)
        if outfit is None:
            print(f"Outfit '{outfit_id}' не найден.")
            return
    else:
        outfit = outfits[-1]

    print(f"Показываю: {outfit['id']}  score: {outfit['score']}")
    for cat, item in zip(outfit["categories"], outfit["items"]):
        print(f"  {cat}: {item}")

    make_collage(outfit).show()


if __name__ == "__main__":
    outfit_id = sys.argv[1] if len(sys.argv) > 1 else None
    view(outfit_id)
