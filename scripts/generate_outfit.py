"""
Генератор outfit из гардероба.

Логика:
  1. Перебирает все комбинации вещей по одной из каждой категории
  2. Считает compatibility score = среднее cosine similarity между парами вещей
  3. Если задан STYLE_PROMPT — добавляет style score: насколько каждая вещь
     соответствует описанию твоего стиля
  4. Итоговый score = compatibility * 0.6 + style * 0.4
  5. Сохраняет топ-N в data/outfits.json
"""

import json
import os
from datetime import datetime
from itertools import combinations, product
from pathlib import Path

import certifi
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npz"
OUTFITS_FILE = DATA_DIR / "outfits.json"

# --- Готовые стили (актуальные тренды 2025-2026) ---
# Выбери один или напиши свой в STYLE_PROMPT ниже.
STYLE_PRESETS = {
    "quiet_luxury":   "quiet luxury, old money aesthetic, neutral tones, black white beige ivory, monochromatic, understated elegant outfit",
    "coquette":       "coquette aesthetic, romantic feminine, soft pastels, lace delicate silhouettes, ballet flats, pearl accessories",
    "soft_academia":  "soft academia, cozy knits, pleated skirts, warm neutrals, oxford shoes, intellectual romantic style",
    "clean_girl":     "clean girl aesthetic, minimal makeup look, white black neutral outfit, sleek polished, simple and chic",
    "mocha_mousse":   "mocha mousse color palette, chocolate brown, warm neutrals, earthy tones, cozy sophisticated outfit",
    "preppy":         "preppy style revival, classic stripes, structured pieces, sharp polished look, collegiate aesthetic",
    "cherry_red":     "cherry red accent, dusty rose, sandy beige, bold color contrast, feminine sophisticated outfit",
    "sport_luxury":   "athleisure quiet luxury mix, sporty silhouettes with elevated fabrics, neutral base with bold color accent, sleek athletic aesthetic, high-end sport style, monochromatic with one vivid pop of color",
}

# --- Твои предпочтения ---
# Вставь ключ из STYLE_PRESETS или напиши свой промпт.
# "" — отключает фильтр стиля.
STYLE_PROMPT = STYLE_PRESETS["sport_luxury"]

# Вес стилевого предпочтения: 0.0 = только совместимость, 1.0 = только стиль
STYLE_WEIGHT = 0.5

OUTFIT_STRUCTURES = [
    ["tops", "pants", "shoes"],
    ["tops", "skirts", "shoes"],
]


def load_embeddings() -> tuple[np.ndarray, list[str], list[str]]:
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    return data["embeddings"], list(data["filenames"]), list(data["categories"])


def load_clip():
    print("Загружаю CLIP модель...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


def get_style_embedding(prompt: str, model, processor) -> np.ndarray:
    """Превращает текстовое описание стиля в вектор."""
    inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_output = model.text_model(**inputs)
        embedding = model.text_projection(text_output.pooler_output)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding[0].numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def get_items_by_category(
    filenames: list[str],
    categories: list[str],
    embeddings: np.ndarray,
) -> dict[str, list[tuple[str, np.ndarray]]]:
    grouped: dict[str, list[tuple[str, np.ndarray]]] = {}
    for filename, category, embedding in zip(filenames, categories, embeddings):
        grouped.setdefault(category, []).append((filename, embedding))
    return grouped


def compatibility_score(combo: list[tuple[str, np.ndarray]]) -> float:
    """Среднее similarity между всеми парами вещей в outfit."""
    pairs = list(combinations(combo, 2))
    if not pairs:
        return 0.0
    return float(np.mean([cosine_similarity(a[1], b[1]) for a, b in pairs]))


def style_score(combo: list[tuple[str, np.ndarray]], style_vec: np.ndarray) -> float:
    """Среднее similarity каждой вещи с текстом стиля."""
    return float(np.mean([cosine_similarity(item[1], style_vec) for item in combo]))


def combined_score(
    combo: list[tuple[str, np.ndarray]],
    style_vec: np.ndarray | None,
    style_weight: float,
) -> float:
    compat = compatibility_score(combo)
    if style_vec is None or style_weight == 0:
        return compat
    style = style_score(combo, style_vec)
    return compat * (1 - style_weight) + style * style_weight


def find_all_outfits(
    grouped: dict[str, list[tuple[str, np.ndarray]]],
    structure: list[str],
    style_vec: np.ndarray | None,
    style_weight: float,
) -> list[tuple[list[str], float]]:
    missing = [cat for cat in structure if cat not in grouped]
    if missing:
        return []

    candidates = list(product(*[grouped[cat] for cat in structure]))
    results = []
    for combo in candidates:
        score = combined_score(list(combo), style_vec, style_weight)
        items = [item[0] for item in combo]
        results.append((items, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def save_outfit(items: list[str], score: float, structure: list[str], style_prompt: str) -> str | None:
    outfits = []
    if OUTFITS_FILE.exists():
        with open(OUTFITS_FILE, "r", encoding="utf-8") as f:
            outfits = json.load(f)

    existing_sets = [set(o["items"]) for o in outfits]
    if set(items) in existing_sets:
        return None

    outfit_id = f"outfit_{len(outfits) + 1:03d}"
    outfit = {
        "id": outfit_id,
        "score": round(score, 4),
        "categories": structure,
        "items": items,
        "style_prompt": style_prompt or None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    outfits.append(outfit)

    with open(OUTFITS_FILE, "w", encoding="utf-8") as f:
        json.dump(outfits, f, ensure_ascii=False, indent=2)

    return outfit_id


def generate(top_k: int = 15):
    embeddings, filenames, categories = load_embeddings()
    grouped = get_items_by_category(filenames, categories, embeddings)

    print("Гардероб по категориям:")
    for cat, items in grouped.items():
        print(f"  {cat}: {len(items)} шт.")

    style_vec = None
    if STYLE_PROMPT:
        model, processor = load_clip()
        style_vec = get_style_embedding(STYLE_PROMPT, model, processor)
        print(f"\nСтиль: \"{STYLE_PROMPT}\" (вес {STYLE_WEIGHT})")

    print()

    all_outfits = []
    for structure in OUTFIT_STRUCTURES:
        for items, score in find_all_outfits(grouped, structure, style_vec, STYLE_WEIGHT):
            all_outfits.append((items, score, structure))

    if not all_outfits:
        print("Не хватает вещей. Нужно: tops + (pants или skirts) + shoes")
        return

    all_outfits.sort(key=lambda x: x[1], reverse=True)

    saved = 0
    for items, score, structure in all_outfits[:top_k]:
        outfit_id = save_outfit(items, score, structure, STYLE_PROMPT)
        if outfit_id is None:
            continue
        saved += 1
        print(f"#{saved} {outfit_id}  score: {score:.4f}")
        for category, item in zip(structure, items):
            print(f"     {category}: {item}")
        print()

    if saved == 0:
        print("Все варианты уже есть в outfits.json")
    else:
        print(f"Сохранено {saved} outfit -> {OUTFITS_FILE}")


if __name__ == "__main__":
    generate(top_k=15)
