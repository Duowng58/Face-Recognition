import os
import cv2
import json
import threading
import queue
import tkinter as tk
from tkinter import messagebox
from insightface.app import FaceAnalysis
from annoy import AnnoyIndex
from face_tracker import Tracker

# ===================== CONFIG =====================
FACE_DATA_DIR = "face_data_1"
ANNOY_INDEX_PATH = os.path.join(FACE_DATA_DIR, "face_index.ann")
MAPPING_PATH = os.path.join(FACE_DATA_DIR, "image_paths.json")

EMBEDDING_DIM = 512
SIM_THRESHOLD = 0.5

CAPTURE_ROOT = "captured_faces"
KNOWN_DIR = os.path.join(CAPTURE_ROOT, "known")
UNKNOWN_DIR = os.path.join(CAPTURE_ROOT, "unknown")

MIN_BBOX_AREA = 10000

RTSP_URL = "rtsp://admin:Ancovn1234@192.168.1.64:554/Streaming/Channels/201/video"

WINDOW_NAME = "Face Recognition Realtime"

DETECT_WIDTH = 1080
DETECT_HEIGHT = 720
# =================================================

os.makedirs(FACE_DATA_DIR, exist_ok=True)
os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ===================== GUI =====================
video_source = None

def use_webcam():
    global video_source
    video_source = 0
    window.quit()

def use_rtsp():
    global video_source
    video_source = RTSP_URL
    window.quit()

window = tk.Tk()
window.title("Chọn nguồn video")

tk.Label(window, text="Face Recognition", font=("Arial", 13)).pack(pady=10)
tk.Button(window, text="📷 Webcam", width=30, command=use_webcam).pack(pady=5)
tk.Button(window, text="📡 RTSP Camera", width=30, command=use_rtsp).pack(pady=10)

window.mainloop()

if video_source is None:
    messagebox.showwarning("Thoát", "Chưa chọn nguồn video")
    exit()

# ===================== GLOBAL =====================
annoy_index = None
idx2name = {}
face_app = None

tracker = Tracker()
trackid_to_name = {}
trackid_saved = set()

annoy_ready = threading.Event()
model_ready = threading.Event()

# ===================== SAVE THREAD =====================
save_queue = queue.Queue()

def save_worker():
    while True:
        frame, path = save_queue.get()
        cv2.imwrite(path, frame)
        save_queue.task_done()

threading.Thread(target=save_worker, daemon=True).start()

# ===================== LOAD =====================
def load_annoy():
    global annoy_index, idx2name
    annoy_index = AnnoyIndex(EMBEDDING_DIM, "angular")

    if os.path.exists(ANNOY_INDEX_PATH):
        annoy_index.load(ANNOY_INDEX_PATH)

    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            idx2name = json.load(f)

    annoy_ready.set()

def load_model():
    global face_app
    face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    face_app.prepare(ctx_id=0)
    model_ready.set()

def recognize(embedding):
    if annoy_index.get_n_items() == 0:
        return "Unknown", 0.0

    idx, dist = annoy_index.get_nns_by_vector(
        embedding, 1, include_distances=True
    )

    sim = 1 - (dist[0] ** 2) / 2
    if sim >= SIM_THRESHOLD:
        return idx2name.get(str(idx[0]), "Unknown"), sim

    return "Unknown", sim

threading.Thread(target=load_annoy, daemon=True).start()
threading.Thread(target=load_model, daemon=True).start()

annoy_ready.wait()
model_ready.wait()

# ===================== CAMERA =====================
if video_source == 0:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"
    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Không mở được video")
    exit()


# ===================== THREADS =====================
latest_frame = None
latest_faces = []
frame_lock = threading.Lock()

def frame_reader():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (1080, 720))

        if video_source == 0:
            frame = cv2.flip(frame, 1)

        with frame_lock:
            latest_frame = frame.copy()

threading.Thread(target=frame_reader, daemon=True).start()


def detect_worker():
    global latest_faces, detect_frame_count,trackid_saved_known,trackid_saved_unknown
    trackid_saved_known = set()
    trackid_saved_unknown = set()

    detect_frame_count = 0
    while True:
        if latest_frame is None:
            continue
        
        detect_frame_count += 1

        if detect_frame_count % 2 != 0:
            continue

        with frame_lock:
            frame_copy = latest_frame.copy()

        rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb)

        rects = []
        embeddings = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)

            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BBOX_AREA:
                continue
            
            rects.append([x1, y1, x2, y2])
            embeddings.append(face.embedding)

        latest_faces = (rects, embeddings)

threading.Thread(target=detect_worker, daemon=True).start()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# ===================== MAIN LOOP =====================
while True:

    if latest_frame is None:
        continue

    with frame_lock:
        frame = latest_frame.copy()

    rects, embeddings = latest_faces if latest_faces else ([], [])

    # TRACK EVERY FRAME
    tracks = tracker.update(rects, classId="face")

    for track in tracks:
        x1, y1, x2, y2, track_id, _ = track

        matched_embedding = None

        for rect, emb in zip(rects, embeddings):
            rx1, ry1, rx2, ry2 = rect

            if abs(x1 - rx1) < 15 and abs(y1 - ry1) < 15:
                matched_embedding = emb
                break

        if matched_embedding is None:
            continue

        name, score = recognize(matched_embedding)
        trackid_to_name[track_id] = (name, score)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        label = f"ID:{track_id} {name} {score:.2f}" \
            if name != "Unknown" else f"ID:{track_id} Unknown"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if name != "Unknown":

            unknown_path = os.path.join(
                UNKNOWN_DIR,
                f"Unknown_ID{track_id}.jpg"
            )
            if os.path.exists(unknown_path):
                os.remove(unknown_path)
          
            if track_id not in trackid_saved_known:
                path = os.path.join(
                    KNOWN_DIR,
                    f"{name}_ID{track_id}.jpg"
                )
                save_queue.put((frame.copy(), path))
                trackid_saved_known.add(track_id)

        # ---------- UNKNOWN ----------
        else:
            if track_id in trackid_saved_known:
                continue

            if track_id not in trackid_saved_unknown:
                path = os.path.join(
                    UNKNOWN_DIR,
                    f"Unknown_ID{track_id}.jpg"
                )
                save_queue.put((frame.copy(), path))
                trackid_saved_unknown.add(track_id)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

