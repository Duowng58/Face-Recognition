import os
import cv2
import json
import numpy as np
import tkinter as tk
import pymongo
from tkinter import filedialog, messagebox, simpledialog
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from annoy import AnnoyIndex

# ================== CONFIG ==================
OUTPUT_FOLDER = "face_data_1"
FRAMES_TO_CAPTURE = 15
SIM_THRESHOLD = 0.5
# ============================================

# ================== GLOBAL ==================
source_type = None   # "webcam" | "folder"
user_name = None
folder_path = None
# ============================================


# ================== GUI ==================
def choose_folder():
    global folder_path
    folder_path = filedialog.askdirectory(title="Chọn thư mục ảnh")
    if folder_path:
        lbl_folder.config(text=f"Đã chọn: {folder_path}")
        entry_name.config(state="disabled")
        entry_name.delete(0, tk.END)
        entry_name.insert(0, "(Tên lấy theo file ảnh)")


def start():
    global source_type, user_name
    if var_webcam.get():
        name = entry_name.get().strip()
        if not name:
            messagebox.showerror("Lỗi", "Vui lòng nhập tên người")
            return
        source_type = "webcam"
        user_name = name

    elif var_folder.get():
        if not folder_path:
            messagebox.showerror("Lỗi", "Chưa chọn thư mục ảnh")
            return
        source_type = "folder"

    else:
        messagebox.showerror("Lỗi", "Vui lòng chọn Webcam hoặc Folder")
        return

    window.destroy()


window = tk.Tk()
window.title("Nạp dữ liệu khuôn mặt")

tk.Label(window, text="Tên người (dùng cho Webcam):").pack()
entry_name = tk.Entry(window, width=40)
entry_name.pack(pady=5)

var_webcam = tk.BooleanVar()
tk.Checkbutton(window, text="Dùng Webcam", variable=var_webcam).pack()

var_folder = tk.BooleanVar()
tk.Checkbutton(window, text="Dùng Folder ảnh (tên file = tên người)",
               variable=var_folder).pack(pady=5)

tk.Button(window, text="Chọn Folder", command=choose_folder).pack()
lbl_folder = tk.Label(window, text="Chưa chọn folder")
lbl_folder.pack(pady=5)

tk.Button(window, text="Bắt đầu", command=start).pack(pady=15)

window.mainloop()

if source_type is None:
    print("[❌] Hủy chương trình")
    exit()

# ================== INIT MODEL ==================
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================== LOAD EXISTING ==================
all_embeddings = []
idx2name = {}

for file in os.listdir(OUTPUT_FOLDER):
    if file.endswith(".npy"):
        name = file.replace(".npy", "")
        data = np.load(os.path.join(OUTPUT_FOLDER, file))
        for vec in data:
            idx2name[len(all_embeddings)] = name
            all_embeddings.append(vec)

# ================== HELPER ==================
def is_duplicate(new_vec):
    for i, old_vec in enumerate(all_embeddings):
        sim = cosine_similarity([new_vec], [old_vec])[0][0]
        if sim > SIM_THRESHOLD:
            return True, idx2name[i]
    return False, None

# ================== WEBCAM MODE ==================
if source_type == "webcam":
    cap = cv2.VideoCapture(0)
    embeddings = []
    count = 0

    print("[🎥] Nhấn 'b' để bắt đầu, 'q' để thoát")

    collecting = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        show = frame.copy()

        if collecting:
            cv2.putText(show, f"Collecting: {len(embeddings)}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            cv2.putText(show, "Nhan 'b' de bat dau",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Webcam", show)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('b'):
            collecting = True

        if collecting and count % 5 == 0 and len(embeddings) < FRAMES_TO_CAPTURE:
            faces = face_app.get(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if faces:
                embeddings.append(faces[0].embedding)

        count += 1
        if len(embeddings) >= FRAMES_TO_CAPTURE:
            break

    cap.release()
    cv2.destroyAllWindows()

    embeddings = np.array(embeddings)
    if len(embeddings) == 0:
        print("[❌] Không thu được khuôn mặt")
        exit()

    dup, match_name = is_duplicate(embeddings[0])
    save_name = user_name

    if dup:
        res = messagebox.askyesno("Trùng khuôn mặt",
                                  f"Trùng với '{match_name}', ghi đè?")
        if not res:
            exit()
        save_name = match_name

    np.save(os.path.join(OUTPUT_FOLDER, f"{save_name}.npy"), embeddings)
    print(f"[✅] Đã lưu {save_name}")

# ================== FOLDER MODE ==================
elif source_type == "folder":
    img_exts = (".jpg", ".png", ".jpeg")
    for file in os.listdir(folder_path):
        if not file.lower().endswith(img_exts):
            continue

        name = os.path.splitext(file)[0]
        img = cv2.imread(os.path.join(folder_path, file))
        if img is None:
            continue

        faces = face_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not faces:
            continue

        #lưu mongodb
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["face"]
        mycol = mydb["student"]
        image_path = os.path.normpath(os.path.join(folder_path, file)).replace("\\", "/")
        mydict = { "name": name, "image": image_path }
        x = mycol.insert_one(mydict)
        print(x.inserted_id)

        emb = faces[0].embedding
        dup, match_name = is_duplicate(emb)
        save_name = x.inserted_id if not dup else match_name

        path = os.path.join(OUTPUT_FOLDER, f"{save_name}.npy")
        if os.path.exists(path):
            old = np.load(path)
            new = np.vstack([old, emb])
        else:
            new = np.array([emb])

        np.save(path, new)
        print(f"[✅] {save_name}: {new.shape[0]} vector")

# ================== REBUILD ANNOY ==================
print("[🔁] Rebuild Annoy Index")
files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".npy")]
dim = np.load(os.path.join(OUTPUT_FOLDER, files[0])).shape[1]

ann = AnnoyIndex(dim, "angular")
idx2name = {}
idx = 0

for f in files:
    name = f.replace(".npy", "")
    data = np.load(os.path.join(OUTPUT_FOLDER, f))
    for v in data:
        ann.add_item(idx, v)
        idx2name[idx] = name
        idx += 1

ann.build(10)
ann.save(os.path.join(OUTPUT_FOLDER, "face_index.ann"))

with open(os.path.join(OUTPUT_FOLDER, "image_paths.json"), "w", encoding="utf-8") as f:
    json.dump(idx2name, f, ensure_ascii=False, indent=2)

print("[✅] Hoàn tất")
