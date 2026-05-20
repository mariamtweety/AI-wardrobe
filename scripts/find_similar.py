"""
Similarity search по гардеробу.

Два режима:
  1. По изображению — найти вещи похожие на заданную
  2. По тексту — найти вещи подходящие под текстовый запрос

Как это работает:
  CLIP обучался сопоставлять изображения и тексты в одном пространстве.
  Это значит эмбеддинг картинки "белая футболка" и эмбеддинг текста
  "white t-shirt" будут близки друг к другу.
  Cosine similarity измеряет угол между двумя векторами:
    1.0 = одинаковые, 0.0 = не связаны, -1.0 = противоположные.
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


DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npz"


def load_embeddings() -> tuple[np.ndarray, list[str], list[str]]:
    """Загружает сохранённые эмбеддинги из файла."""
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    embeddings = data["embeddings"]
    filenames = list(data["filenames"])
    categories = list(data["categories"])
    return embeddings, filenames, categories


def load_clip():
    print("Загружаю CLIP модель...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Вычисляет cosine similarity между двумя векторами."""
    return float(np.dot(a, b))  # векторы уже нормализованы при сохранении


def search_by_image(query_filename: str, top_k: int = 3):
    """
    Находит вещи похожие на заданную по имени файла.
    Сама вещь в результаты не попадает.
    """
    embeddings, filenames, categories = load_embeddings()

    if query_filename not in filenames:
        print(f"Файл '{query_filename}' не найден в эмбеддингах.")
        print(f"Доступные файлы: {filenames}")
        return

    query_idx = filenames.index(query_filename)
    query_embedding = embeddings[query_idx]

    results = []
    for i, (filename, category) in enumerate(zip(filenames, categories)):
        if i == query_idx:
            continue  # пропускаем саму вещь
        similarity = cosine_similarity(query_embedding, embeddings[i])
        results.append((filename, category, similarity))

    results.sort(key=lambda x: x[2], reverse=True)

    print(f"\nПохожие на '{query_filename}':")
    for filename, category, score in results[:top_k]:
        print(f"  {filename} [{category}]  similarity: {score:.3f}")


def search_by_text(text_query: str, top_k: int = 3):
    """
    Находит вещи подходящие под текстовый запрос.
    CLIP сравнивает текст с изображениями в одном пространстве.
    """
    embeddings, filenames, categories = load_embeddings()
    model, processor = load_clip()

    # получаем эмбеддинг текста
    inputs = processor(text=[text_query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_output = model.text_model(**inputs)
        text_embedding = model.text_projection(text_output.pooler_output)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    text_vec = text_embedding[0].numpy()

    # logit_scale — температурный коэффициент из оригинального CLIP
    # усиливает различия между вещами, делая вывод нагляднее
    logit_scale = model.logit_scale.exp().item()

    raw_scores = np.array([
        cosine_similarity(text_vec, embeddings[i]) for i in range(len(filenames))
    ])
    scaled = raw_scores * logit_scale
    # softmax превращает числа в вероятности (сумма = 100%)
    probs = np.exp(scaled) / np.exp(scaled).sum()

    results = list(zip(filenames, categories, probs))
    results.sort(key=lambda x: x[2], reverse=True)

    print(f"\nРезультаты для запроса '{text_query}':")
    for filename, category, prob in results[:top_k]:
        print(f"  {filename} [{category}]  {prob * 100:.1f}%")


if __name__ == "__main__":
    embeddings, filenames, categories = load_embeddings()

    print("=" * 50)
    print("Гардероб:")
    for filename, category in zip(filenames, categories):
        print(f"  {filename} [{category}]")

    print("\n" + "=" * 50)
    print("Similarity между всеми вещами:\n")
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            score = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  {filenames[i]}  <->  {filenames[j]}")
            print(f"  similarity: {score:.3f}\n")

    print("=" * 50)
    print("Поиск по тексту:\n")
    for query in ["white t-shirt", "black pants", "casual outfit"]:
        search_by_text(query)
