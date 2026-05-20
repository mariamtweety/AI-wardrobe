"""
Собирает metadata о всех вещах в гардеробе и сохраняет в data/wardrobe.json.

Запускать ПОСЛЕ classify_category.py — скрипт ищет PNG по папкам категорий.

Зачем metadata: когда придёт время делать outfit generator и рекомендации,
нам нужно быстро знать что есть в гардеробе, какой категории, когда добавлено.
JSON — простой формат для этого этапа, без базы данных.
"""

import json
from datetime import date
from pathlib import Path

from PIL import Image


PREPROCESSED_DIR = Path(__file__).parent.parent / "preprocessed"
DATA_DIR = Path(__file__).parent.parent / "data"
METADATA_FILE = DATA_DIR / "wardrobe.json"

CATEGORIES = ["tops", "pants", "shoes", "outerwear", "accessories"]


def build_item_record(image_path: Path, category: str) -> dict:
    """Создаёт запись для одной вещи."""
    with Image.open(image_path) as img:
        width, height = img.size

    # stem — имя файла без расширения (например, "IMG_001")
    item_id = image_path.stem

    return {
        "id": item_id,
        "filename": image_path.name,
        "category": category,
        "size": {"width": width, "height": height},
        "added_date": str(date.today()),
    }


def build_metadata():
    items = []

    for category in CATEGORIES:
        category_dir = PREPROCESSED_DIR / category
        if not category_dir.exists():
            continue

        png_files = list(category_dir.glob("*.png"))
        for image_path in png_files:
            record = build_item_record(image_path, category)
            items.append(record)
            print(f"  + {record['filename']} [{category}]")

    if not items:
        print("Нет вещей в папках категорий.")
        print("Сначала запусти scripts/classify_category.py")
        return

    # сохраняем с отступами чтобы JSON был читаемым
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено {len(items)} вещей -> {METADATA_FILE}")


if __name__ == "__main__":
    build_metadata()
