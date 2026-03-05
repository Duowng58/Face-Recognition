import os
import cv2
import json
import shutil
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from annoy import AnnoyIndex
from insightface.app import FaceAnalysis

# ===================== CONFIG =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UNKNOWN_DIR = os.path.join(BASE_DIR, "captured_faces", "unknown")
KNOWN_DIR   = os.path.join(BASE_DIR, "captured_faces", "known")

FACE_DATA_DIR = os.path.join(BASE_DIR, "face_data")
ANNOY_INDEX_PATH = os.path.join(FACE_DATA_DIR, "face_index.ann")
MAPPING_PATH     = os.path.join(FACE_DATA_DIR, "image_paths.json")

EMBEDDING_DIM = 512
N_TREES = 10

os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(FACE_DATA_DIR, exist_ok=True)

# ===================== FACE MODEL =====================
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ===================== HELPERS =====================
def extract_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)

    if not faces:
        return None

    return faces[0].embedding


def rebuild_annoy_index():
    print("[🔄] Rebuilding Annoy index...")

    index = AnnoyIndex(EMBEDDING_DIM, "angular")
    mapping = {}

    idx = 0
    for file in os.listdir(KNOWN_DIR):
        if not file.lower().endswith((".jpg", ".png")):
            continue

        path = os.path.join(KNOWN_DIR, file)
        emb = extract_embedding(path)
        if emb is None:
            continue

        name = file.split("_")[0]

        index.add_item(idx, emb)
        mapping[str(idx)] = name
        idx += 1

    if idx == 0:
        print("[⚠️] No known faces, skip building")
        return

    index.build(N_TREES)
    index.save(ANNOY_INDEX_PATH)

    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)

    print(f"[✅] Annoy rebuilt: {idx} faces")


# ===================== GUI =====================
root = tk.Tk()
root.title("Gán tên khuôn mặt Unknown")
root.geometry("600x500")

unknown_files = []

def load_unknown_list():
    global unknown_files
    listbox.delete(0, tk.END)

    unknown_files = [
        f for f in os.listdir(UNKNOWN_DIR)
        if f.lower().endswith((".jpg", ".png"))
    ]

    for f in unknown_files:
        listbox.insert(tk.END, f)


def show_image(event=None):
    if not listbox.curselection():
        return

    idx = listbox.curselection()[0]
    img_path = os.path.join(UNKNOWN_DIR, unknown_files[idx])

    img = Image.open(img_path).resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)

    img_label.config(image=img_tk)
    img_label.image = img_tk


def assign_name():
    if not listbox.curselection():
        messagebox.showwarning("Lỗi", "Chưa chọn ảnh")
        return

    name = entry.get().strip()
    if not name:
        messagebox.showwarning("Lỗi", "Chưa nhập tên")
        return

    idx = listbox.curselection()[0]
    filename = unknown_files[idx]

    src = os.path.join(UNKNOWN_DIR, filename)
    dst = os.path.join(KNOWN_DIR, f"{name}_{filename}")

    emb = extract_embedding(src)
    if emb is None:
        messagebox.showerror("Lỗi", "Không phát hiện khuôn mặt")
        return

    shutil.move(src, dst)
    rebuild_annoy_index()
    load_unknown_list()
    img_label.config(image="")
    entry.delete(0, tk.END)
    messagebox.showinfo("OK", f"Đã gán tên: {name}")

# ===================== UI LAYOUT =====================
listbox = tk.Listbox(root, width=40)
listbox.pack(pady=10)
listbox.bind("<<ListboxSelect>>", show_image)

img_label = tk.Label(root)
img_label.pack(pady=10)

entry = tk.Entry(root, width=30)
entry.pack(pady=5)

btn = tk.Button(root, text="Gán tên", command=assign_name)
btn.pack(pady=10)

load_unknown_list()
root.mainloop()