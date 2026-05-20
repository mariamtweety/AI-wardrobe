# AI Digital Wardrobe

## Описание проекта

AI Digital Wardrobe — это веб-приложение, которое:

* хранит цифровой гардероб пользователя,
* анализирует одежду,
* предлагает готовые образы,
* обучается на лайках/дизлайках пользователя,
* постепенно начинает понимать стиль пользователя.

---

# Главная цель

Создать систему:

```text
гардероб → понимание вещей → рекомендации → персональный вкус
```

---

# Основной user flow

## 1. Загрузка одежды

Пользователь загружает:

* футболки,
* худи,
* обувь,
* брюки,
* верхнюю одежду,
* аксессуары.

---

## 2. Обработка изображений

Система автоматически:

* удаляет фон,
* обрезает изображение,
* делает preprocessing,
* подготавливает изображение для ML.

---

## 3. Анализ одежды

Система:

* получает embedding вещи через CLIP,
* определяет similarity между вещами,
* анализирует визуальный стиль.

---

## 4. Генерация outfit

Система:

* комбинирует вещи,
* предлагает готовые образы,
* подбирает совместимые сочетания.

---

## 5. Feedback system

Пользователь оценивает outfit:

* ❤️ нравится
* ❌ не нравится

---

## 6. Обучение системы

Recommendation system постепенно:

* понимает вкус пользователя,
* обучается на likes/dislikes,
* улучшает рекомендации.

---

# MVP функционал

---

# МОДУЛЬ 1 — Digital wardrobe

## Возможности

* загрузка одежды,
* хранение изображений,
* категории вещей,
* basic metadata.

---

## Категории

Например:

* tops
* pants
* shoes
* outerwear
* accessories

---

# МОДУЛЬ 2 — Image preprocessing

## Pipeline

```text
image
→ background removal
→ crop
→ resize
→ padding
→ preprocessing
```

---

## Цель

Получить:

* чистые изображения,
* одинаковый формат,
* стабильные данные для embeddings.

---

# МОДУЛЬ 3 — Embedding system

## Использование CLIP

Система:

* получает embedding каждой вещи,
* сохраняет embeddings,
* сравнивает вещи через cosine similarity.

---

## Возможности embeddings

### Similarity search

```text
найти похожие вещи
```

---

### Style understanding

Например:

* minimal
* monochrome
* streetwear
* old money

---

### Compatibility understanding

Система пытается понимать:

* какие вещи визуально сочетаются,
* какие aesthetic близки друг другу.

---

# МОДУЛЬ 4 — Outfit generator

## Логика

Система:

* выбирает верх,
* подбирает низ,
* подбирает обувь,
* собирает outfit.

---

## Первый этап

Без генерации новых изображений.

Только:

* комбинирование существующих вещей,
* ranking outfit по score.

---

# МОДУЛЬ 5 — Recommendation system

## Источник обучения

```text
likes / dislikes
```

---

## Цель

Recommendation system должна:

* понимать вкус пользователя,
* определять preferred aesthetics,
* улучшать рекомендации.

---

# МОДУЛЬ 6 — Swipe system

## Механика

* свайп вправо → нравится
* свайп влево → не нравится

---

## Зачем это нужно

Swipe system используется для:

* UX,
* сбора dataset,
* обучения recommendation model.

---

# МОДУЛЬ 7 — User profile

## Возможности

Пользователь может:

* хранить гардероб,
* сохранять outfits,
* смотреть liked outfits.

---

# Future features

## Weather integration

Например:

```text
outfit for rainy Berlin
```

---

## Occasion-based outfits

Например:

* office
* university
* dinner
* travel

---

## Text prompts

Например:

```text
quiet luxury outfit
```

---

## AI stylist chat

Например:

```text
что надеть сегодня?
```

---

## Virtual try-on

AI-примерка одежды.

---

# Архитектура проекта

## Frontend

* Next.js
* Tailwind CSS

---

## Backend

* Python
* FastAPI

---

## ML stack

* PyTorch
* transformers
* CLIP

---

## Database

* PostgreSQL

---

## Storage

* S3-compatible storage

---

# Этапы разработки

## ЭТАП 1

Image preprocessing

---

## ЭТАП 2

CLIP embeddings

---

## ЭТАП 3

Similarity experiments

---

## ЭТАП 4

Outfit generator

---

## ЭТАП 5

Like/dislike dataset

---

## ЭТАП 6

Recommendation system

---

## ЭТАП 7

Web application

---

## ЭТАП 8

Advanced AI features

---

# Главная техническая идея

Не:

* генерировать новую одежду,

А:

* понимать совместимость существующих вещей.

---

# Главная AI идея

Система должна:

* обучаться на вкусе пользователя,
* постепенно становиться персональным AI stylist.
