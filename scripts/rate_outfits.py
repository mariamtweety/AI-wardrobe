"""
Просмотр и оценка аутфитов.

Управление:
  A / D — предыдущий / следующий аутфит
  W     — лайк ❤
  S     — дизлайк ✗
  Q     — выйти
"""

import json
from datetime import datetime
from pathlib import Path

import tkinter as tk
from PIL import Image, ImageTk


DATA_DIR = Path(__file__).parent.parent / "data"
PREPROCESSED_DIR = Path(__file__).parent.parent / "preprocessed"
OUTFITS_FILE = DATA_DIR / "outfits.json"
FEEDBACK_FILE = DATA_DIR / "feedback.json"

CATEGORIES = ["tops", "pants", "skirts", "shoes", "outerwear", "accessories"]
ITEM_SIZE = 280
GAP = 16


def load_outfits() -> list[dict]:
    with open(OUTFITS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_feedback() -> list[dict]:
    if not FEEDBACK_FILE.exists():
        return []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_feedback(feedback: list[dict]):
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)


def find_image(filename: str) -> Path | None:
    for category in CATEGORIES:
        path = PREPROCESSED_DIR / category / filename
        if path.exists():
            return path
    return None


def make_collage(outfit: dict) -> Image.Image:
    images = []
    for filename in outfit["items"]:
        path = find_image(filename)
        if path is None:
            continue
        img = Image.open(path).convert("RGBA")
        canvas = Image.new("RGBA", (ITEM_SIZE, ITEM_SIZE), (245, 245, 245, 255))
        img.thumbnail((ITEM_SIZE, ITEM_SIZE))
        x = (ITEM_SIZE - img.width) // 2
        y = (ITEM_SIZE - img.height) // 2
        canvas.paste(img, (x, y), mask=img)
        images.append(canvas)

    total_width = ITEM_SIZE * len(images) + GAP * (len(images) - 1)
    collage = Image.new("RGB", (total_width, ITEM_SIZE), (255, 255, 255))
    for i, img in enumerate(images):
        collage.paste(img.convert("RGB"), (i * (ITEM_SIZE + GAP), 0))
    return collage


def get_rating(feedback: list[dict], outfit_id: str) -> str | None:
    for f in feedback:
        if f["outfit_id"] == outfit_id:
            return f["rating"]
    return None


def rate():
    outfits = load_outfits()
    if not outfits:
        print("Нет аутфитов. Запусти generate_outfit.py")
        return

    feedback = load_feedback()
    idx = [0]  # список чтобы менять внутри вложенных функций

    root = tk.Tk()
    root.title("AI Wardrobe")
    root.configure(bg="#ffffff")
    root.resizable(False, False)

    img_label = tk.Label(root, bg="#ffffff")
    img_label.pack(padx=20, pady=(20, 8))

    info_label = tk.Label(root, bg="#ffffff", font=("SF Pro", 13), fg="#333333")
    info_label.pack()

    rating_label = tk.Label(root, bg="#ffffff", font=("SF Pro", 18))
    rating_label.pack(pady=(4, 4))

    hint = tk.Label(
        root,
        text="A ◀  D ▶     W ❤  S ✗     Q выйти",
        bg="#ffffff",
        font=("SF Pro", 11),
        fg="#999999",
    )
    hint.pack(pady=(0, 16))

    tk_image = [None]  # держим ссылку чтобы не собрал GC

    def render():
        outfit = outfits[idx[0]]
        collage = make_collage(outfit)
        tk_image[0] = ImageTk.PhotoImage(collage)
        img_label.config(image=tk_image[0])

        rating = get_rating(feedback, outfit["id"])
        rating_label.config(
            text="❤" if rating == "like" else ("✗" if rating == "dislike" else ""),
            fg="#e05c5c" if rating == "like" else "#555555",
        )
        info_label.config(
            text=f"{outfit['id']}  •  score {outfit['score']}  •  {idx[0]+1}/{len(outfits)}"
        )
        root.update_idletasks()
        root.geometry("")  # авторазмер

    def rate_current(rating: str):
        outfit = outfits[idx[0]]
        # удаляем предыдущую оценку если была
        nonlocal feedback
        feedback = [f for f in feedback if f["outfit_id"] != outfit["id"]]
        feedback.append({
            "outfit_id": outfit["id"],
            "rating": rating,
            "items": outfit["items"],
            "categories": outfit["categories"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        save_feedback(feedback)
        render()

    def on_key(event):
        key = event.keysym.lower()
        if key == "d":
            idx[0] = (idx[0] + 1) % len(outfits)
            render()
        elif key == "a":
            idx[0] = (idx[0] - 1) % len(outfits)
            render()
        elif key == "w":
            rate_current("like")
        elif key == "s":
            rate_current("dislike")
        elif key == "q":
            root.destroy()

    root.bind("<Key>", on_key)
    render()
    root.mainloop()

    likes = sum(1 for f in feedback if f["rating"] == "like")
    dislikes = sum(1 for f in feedback if f["rating"] == "dislike")
    print(f"Оценок: {likes} ❤  {dislikes} ✗")


if __name__ == "__main__":
    rate()
