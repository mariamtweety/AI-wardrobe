# AI Digital Wardrobe — план разработки

## Текущий статус

| Этап | Статус |
|------|--------|
| Этап 1 — Image preprocessing | ✅ Готово |
| Этап 2 — CLIP embeddings | ⬜ Не начат |
| Этап 3 — Similarity experiments | ⬜ Не начат |
| Этап 4 — Outfit generator | ⬜ Не начат |
| Этап 5 — Like/dislike dataset | ⬜ Не начат |
| Этап 6 — Recommendation system | ⬜ Не начат |
| Этап 7 — Web application | ⬜ Не начат |
| Этап 8 — Advanced AI features | ⬜ Не начат |

---

## Этап 1 — Image preprocessing

**Цель:** получить чистые, стандартизированные изображения для ML-моделей.

### Что уже сделано

- `scripts/remove_background.py` — удаление фона через rembg, сохраняет PNG с прозрачностью в `processed/`
- `scripts/preprocess_images.py` — crop → resize → padding до 224×224, сохраняет в `preprocessed/`

### Что осталось сделать

#### 1.1 Классификация по категориям

Скрипт: `scripts/classify_category.py`

Использовать CLIP zero-shot classification — без обучения, просто сравнить эмбеддинг изображения с текстовыми метками:

```python
# Логика
categories = ["t-shirt", "pants", "shoes", "jacket", "dress", "hoodie", "accessories"]
# CLIP сравнивает изображение с каждым текстом и выбирает наиболее похожий
```

Результат: каждое изображение получает папку-категорию:
```
preprocessed/
├── tops/
├── pants/
├── shoes/
├── outerwear/
└── accessories/
```

#### 1.2 Metadata файл

Скрипт: `scripts/build_metadata.py`

Сохранять JSON с базовой информацией о каждой вещи:

```json
{
  "id": "IMG_001",
  "filename": "IMG_001.png",
  "category": "tops",
  "original_size": [1080, 1920],
  "processed_size": [224, 224],
  "added_date": "2026-05-19"
}
```

Файл: `data/wardrobe.json`

**Почему JSON а не база данных сейчас:** на этапе экспериментов JSON достаточно, быстро читается и редактируется. База данных — на этапе веб-приложения.

---

## Этап 2 — CLIP embeddings

**Цель:** превратить каждое изображение в вектор (embedding) — числовое представление, которое CLIP понимает как "смысл" изображения.

**Почему CLIP:** модель обучена на парах изображение-текст, поэтому понимает визуальный стиль, цвет, фактуру. Можно искать вещи по тексту ("белая рубашка") без ручной разметки.

### Структура

```
scripts/
└── generate_embeddings.py

data/
└── embeddings.npz   # numpy-файл с векторами
```

### Реализация

```python
# Библиотека: transformers (HuggingFace)
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Для каждого изображения:
# 1. processor подготавливает изображение (resize, normalize)
# 2. model.get_image_features() возвращает вектор 512 чисел
# 3. normalize вектор (для cosine similarity)
# 4. сохранить в embeddings.npz вместе с именами файлов
```

### Зависимости

```
transformers
torch
```

### Что получим

Файл `data/embeddings.npz`:
- ключ `filenames` — список имён файлов
- ключ `embeddings` — матрица N × 512 (N вещей, 512 чисел каждая)

---

## Этап 3 — Similarity experiments

**Цель:** научиться находить похожие вещи и понимать визуальный стиль.

### 3.1 Поиск похожих вещей

Скрипт: `scripts/find_similar.py`

```python
# Cosine similarity — мера схожести двух векторов
# Значение от -1 до 1, чем ближе к 1 — тем похожее вещи
from sklearn.metrics.pairwise import cosine_similarity

# Пример: найти топ-3 похожих вещи для заданного изображения
similarities = cosine_similarity([query_embedding], all_embeddings)
top_3 = similarities.argsort()[-3:][::-1]
```

### 3.2 Поиск по тексту

CLIP умеет сравнивать изображения и текст в одном пространстве:

```python
# "найди мне белую рубашку"
text_inputs = processor(text=["white shirt"], return_tensors="pt")
text_embedding = model.get_text_features(**text_inputs)

# сравниваем text_embedding со всеми image embeddings
# самое похожее изображение — ответ
```

### 3.3 Style clustering

Скрипт: `scripts/cluster_styles.py`

```python
# K-Means кластеризация эмбеддингов
# Вещи с похожими эмбеддингами попадают в один кластер
# Кластеры ≈ визуальные стили (minimal, streetwear, formal...)
from sklearn.cluster import KMeans
```

### Зависимости

```
scikit-learn
```

---

## Этап 4 — Outfit generator

**Цель:** автоматически подбирать совместимые вещи из гардероба.

**Принцип:** не генерировать новую одежду, а комбинировать существующую. Совместимость = близость эмбеддингов вещей разных категорий.

### Логика

```python
# Алгоритм:
# 1. Взять одну вещь (например, брюки)
# 2. Найти топ вещей из другой категории (топы) с наибольшей similarity
# 3. Добавить обувь по тому же принципу
# 4. Outfit = [топ, брюки, обувь]

def generate_outfit(anchor_item, wardrobe_embeddings):
    top = find_most_compatible(anchor_item, category="tops")
    bottom = find_most_compatible(anchor_item, category="pants")
    shoes = find_most_compatible(anchor_item, category="shoes")
    return Outfit(top, bottom, shoes)
```

### Scoring

Outfit получает score = среднее cosine similarity между всеми парами вещей. Чем выше score — тем лучше сочетание.

### Скрипт

`scripts/generate_outfit.py`

### Результат

```json
{
  "outfit_id": "outfit_001",
  "score": 0.87,
  "items": ["hoodie_01.png", "pants_02.png", "shoes_03.png"]
}
```

---

## Этап 5 — Like/dislike dataset

**Цель:** собрать данные о вкусе пользователя для обучения рекомендаций.

### Механика

Простой CLI (потом заменится на swipe UI):

```bash
python scripts/rate_outfits.py
# Показывает outfit, спрашивает: [L]ike / [D]islike / [S]kip
```

### Структура данных

```json
// data/feedback.json
[
  {
    "outfit_id": "outfit_001",
    "rating": "like",
    "timestamp": "2026-05-19T14:30:00",
    "items": ["hoodie_01.png", "pants_02.png", "shoes_03.png"]
  }
]
```

---

## Этап 6 — Recommendation system

**Цель:** использовать лайки/дизлайки чтобы улучшить рекомендации.

### Подход

**User preference vector** — среднее эмбеддингов понравившихся вещей. Это вектор "вкуса пользователя".

```python
# liked_embeddings — эмбеддинги вещей из liked outfits
user_taste = liked_embeddings.mean(axis=0)

# Теперь ищем вещи, близкие к user_taste
# Это и есть персональная рекомендация
recommendations = find_similar(user_taste, all_embeddings, top_k=5)
```

Этот подход называется **content-based filtering** — простой, не требует много данных.

### Скрипт

`scripts/recommend.py`

---

## Этап 7 — Web application

**Цель:** упаковать всё в веб-приложение с нормальным UI.

### Архитектура

```
Frontend: Next.js + Tailwind CSS
Backend:  Python + FastAPI
Database: PostgreSQL
Storage:  S3-compatible (например, MinIO локально)
```

### API endpoints (основные)

```
POST /items/upload       — загрузить вещь
GET  /items              — список гардероба
GET  /outfits/generate   — сгенерировать outfit
POST /outfits/{id}/like  — лайкнуть outfit
POST /outfits/{id}/dislike
GET  /recommendations    — персональные рекомендации
```

### Порядок разработки

1. FastAPI backend с базовыми эндпоинтами
2. PostgreSQL схема (items, outfits, feedback)
3. Next.js frontend — загрузка вещей, просмотр гардероба
4. Swipe UI для outfit rating
5. Страница рекомендаций

---

## Этап 8 — Advanced AI features

Добавлять только после того как работает базовый flow.

| Фича | Реализация |
|------|-----------|
| Поиск по тексту ("quiet luxury outfit") | CLIP text embeddings → similarity search |
| Weather-based outfits | OpenWeatherMap API → фильтр категорий |
| Occasion-based ("на работу", "в кафе") | CLIP text prompt как фильтр |
| AI stylist чат | Claude API + контекст гардероба |
| Virtual try-on | Отдельная диффузионная модель (например, IDM-VTON) |

---

## Структура проекта (финальная)

```
AI-wardrobe/
├── raw/                        # исходные фото
├── processed/                  # после background removal
├── preprocessed/               # после crop + resize + padding
│   ├── tops/
│   ├── pants/
│   ├── shoes/
│   ├── outerwear/
│   └── accessories/
├── data/
│   ├── wardrobe.json           # metadata вещей
│   ├── embeddings.npz          # CLIP vectors
│   └── feedback.json           # likes/dislikes
├── scripts/
│   ├── remove_background.py    # ✅ готово
│   ├── preprocess_images.py    # ✅ готово
│   ├── classify_category.py    # следующий шаг
│   ├── build_metadata.py
│   ├── generate_embeddings.py
│   ├── find_similar.py
│   ├── cluster_styles.py
│   ├── generate_outfit.py
│   ├── rate_outfits.py
│   └── recommend.py
├── backend/                    # FastAPI (этап 7)
├── frontend/                   # Next.js (этап 7)
├── requirements.txt
├── README.md
└── PLAN.md
```

---

## Следующий шаг прямо сейчас

**Этап 1.1** — `scripts/classify_category.py`

Это логично делать следующим: после preprocessing каждая вещь должна знать свою категорию, прежде чем мы начнём генерировать outfit.
