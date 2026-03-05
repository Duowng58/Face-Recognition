import os
import cv2
import json
import time
import threading
import queue
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from insightface.app import FaceAnalysis
from annoy import AnnoyIndex
from face_tracker import Tracker

# ===================== CONFIG =====================
FACE_DATA_DIR = "face_data_1"
ANNOY_INDEX_PATH = os.path.join(FACE_DATA_DIR, "face_index.ann")
MAPPING_PATH = os.path.join(FACE_DATA_DIR, "image_paths.json")

EMBEDDING_DIM  = 512
SIM_THRESHOLD  = 0.5

CAPTURE_ROOT      = "captured_faces"
KNOWN_DIR         = os.path.join(CAPTURE_ROOT, "known")
UNKNOWN_DIR       = os.path.join(CAPTURE_ROOT, "unknown")
KNOWN_VIDEO_DIR   = os.path.join(CAPTURE_ROOT, "known")    
UNKNOWN_VIDEO_DIR = os.path.join(CAPTURE_ROOT, "unknown")  

MIN_BBOX_AREA     = 30000
FACE_CROP_PADDING = 20

RTSP_URL    = "rtsp://admin:Ancovn1234@192.168.1.64:554/Streaming/Channels/201/video"
WINDOW_NAME = "Face Recognition Realtime"

DETECT_WIDTH  = 1080
DETECT_HEIGHT = 960

VIDEO_FPS    = 15
VIDEO_FOURCC = "mp4v"
# ==================================================

os.makedirs(FACE_DATA_DIR,    exist_ok=True)
os.makedirs(KNOWN_DIR,        exist_ok=True)
os.makedirs(UNKNOWN_DIR,      exist_ok=True)


# ===================== Chọn nguồn video =====================
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
tk.Button(window, text="📷 Webcam",      width=30, command=use_webcam).pack(pady=5)
tk.Button(window, text="📡 RTSP Camera", width=30, command=use_rtsp).pack(pady=10)
window.mainloop()

if video_source is None:
    messagebox.showwarning("Thoát", "Chưa chọn nguồn video")
    exit()

# ===================== GLOBAL STATE =====================
annoy_index = None
idx2name    = {}
face_app    = None

tracker         = Tracker()
trackid_to_name = {}
trackid_saved_known   = set()   
trackid_saved_unknown = set()  

annoy_ready = threading.Event()
model_ready = threading.Event()

frame_queue  = queue.Queue(maxsize=3)
result_queue = queue.Queue(maxsize=3)
save_queue   = queue.Queue()

video_write_queue = queue.Queue()

running = threading.Event()
running.set()


# ===================== SAVE WORKER (ảnh thumbnail) =====================
def save_worker():
    while True:
        frame, path = save_queue.get()
        cv2.imwrite(path, frame)
        save_queue.task_done()

threading.Thread(target=save_worker, daemon=True).start()


# ===================== VIDEO WRITE WORKER =====================
def video_write_worker():
    writers   = {}
    vid_paths = {}

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)

    def _open_writer(track_id, frame, path):
        h, w = frame.shape[:2]
        writers[track_id]   = cv2.VideoWriter(path, fourcc, VIDEO_FPS, (w, h))
        vid_paths[track_id] = path

    while True:
        try:
            track_id, frame, label = video_write_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if label == "__stop__":
            for tid, w in writers.items():
                w.release()
                print(f"[VideoWriter] Đóng (giữ file) ID{tid}: {vid_paths.get(tid,'')}")
            writers.clear()
            vid_paths.clear()
            video_write_queue.task_done()
            break

        if label.startswith("__promote__:"):
            name = label.split(":", 1)[1]
            if track_id in writers:
                writers[track_id].release()
                del writers[track_id]

            old_path = vid_paths.pop(track_id, None)
            if old_path and os.path.exists(old_path):
                new_path = os.path.join(KNOWN_VIDEO_DIR, f"{name}_ID{track_id}.mp4")
                os.rename(old_path, new_path)
                print(f"[VideoWriter] Promote ID{track_id}: {old_path} → {new_path}")
                vid_paths[track_id] = new_path
            video_write_queue.task_done()
            continue

        # ── Frame bình thường ──
        if frame is None:
            video_write_queue.task_done()
            continue

        if track_id not in writers:
            if label.startswith("known:"):
                name     = label.split(":", 1)[1]
                out_path = os.path.join(KNOWN_VIDEO_DIR,   f"{name}_ID{track_id}.mp4")
            else:  
                out_path = os.path.join(UNKNOWN_VIDEO_DIR, f"Unknown_ID{track_id}.mp4")

            existing = vid_paths.get(track_id)
            if existing and os.path.exists(existing):
                out_path = existing

            _open_writer(track_id, frame, out_path)
            print(f"[VideoWriter] Bắt đầu ghi ID{track_id}: {out_path}")

        writers[track_id].write(frame)
        video_write_queue.task_done()

threading.Thread(target=video_write_worker, daemon=True, name="T-VideoWrite").start()


# ===================== LOAD ANNOY + MODEL =====================
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
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider"]
    )
    face_app.prepare(ctx_id=0)
    model_ready.set()

threading.Thread(target=load_annoy, daemon=True).start()
threading.Thread(target=load_model, daemon=True).start()

annoy_ready.wait()
model_ready.wait()
print("✅ Model và Annoy index đã sẵn sàng")


# ===================== MỞ CAMERA =====================
if video_source == 0:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"
    )
    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Không mở được video source")
    exit()


# ===================== RECOGNIZE HELPER =====================
def recognize(embedding):
    """Trả về (name, similarity_score)"""
    if annoy_index.get_n_items() == 0:
        return "Unknown", 0.0
    idx, dist = annoy_index.get_nns_by_vector(embedding, 1, include_distances=True)
    sim = 1 - (dist[0] ** 2) / 2
    if sim >= SIM_THRESHOLD:
        return idx2name.get(str(idx[0]), "Unknown"), sim
    return "Unknown", sim


# ===================== CAPTURE FRAMES =====================
def capture_frames():
    try:
        while running.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (DETECT_WIDTH, DETECT_HEIGHT))

            if video_source == 0:
                frame = cv2.flip(frame, 1)

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass

            frame_queue.put(frame)

    except Exception as e:
        print(f"[capture_frames] Lỗi: {e}")
    finally:
        print("capture_frames dừng")


# ===================== CROP FACE HELPER =====================
def crop_face(frame, x1, y1, x2, y2, padding=FACE_CROP_PADDING):
    h, w = frame.shape[:2]
    cx1 = max(0, x1 - padding)
    cy1 = max(0, y1 - padding)
    cx2 = min(w, x2 + padding)
    cy2 = min(h, y2 + padding)
    return frame[cy1:cy2, cx1:cx2]


# ===================== PROCESS FRAMES =====================
def process_frames():
    global trackid_to_name, trackid_saved_known, trackid_saved_unknown

    # track_id → nhãn hiện tại đang ghi video ("unknown" hoặc "known:{name}")
    recording_label: dict[int, str] = {}

    try:
        while running.is_set():
            try:
                frame = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # ── Detect ──
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_app.get(rgb)

            rects      = []
            embeddings = []
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                if (x2 - x1) * (y2 - y1) < MIN_BBOX_AREA:
                    continue
                rects.append([x1, y1, x2, y2])
                embeddings.append(face.embedding)

            # ── Track ──
            tracks = tracker.update(rects, classId="face")

            # ── Map embedding → track_id ──
            trackid_to_emb_now = {}
            for rect, emb in zip(rects, embeddings):
                rx1, ry1, rx2, ry2 = rect
                rcx = (rx1 + rx2) / 2
                rcy = (ry1 + ry2) / 2

                best_id, best_dist = None, float("inf")
                for x1, y1, x2, y2, tid, _ in tracks:
                    d = ((rcx - (x1 + x2) / 2) ** 2 + (rcy - (y1 + y2) / 2) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        best_id   = tid

                if best_id is not None and best_dist < 150:
                    trackid_to_emb_now[best_id] = emb

            # ── Recognize + vẽ ──
            annotated = frame.copy()

            for x1, y1, x2, y2, track_id, _ in tracks:
                if track_id in trackid_to_emb_now:
                    name, score = recognize(trackid_to_emb_now[track_id])
                    trackid_to_name[track_id] = (name, score)
                else:
                    name, score = trackid_to_name.get(track_id, ("Unknown", 0.0))

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = (
                    f"ID:{track_id} {name} {score:.2f}"
                    if name != "Unknown"
                    else f"ID:{track_id} Unknown"
                )
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

                face_crop = crop_face(frame, x1, y1, x2, y2)

                # ── Xử lý known ──
                if name != "Unknown":
                    if track_id not in trackid_saved_known:
                        unknown_thumb = os.path.join(UNKNOWN_DIR, f"Unknown_ID{track_id}.jpg")
                        if os.path.exists(unknown_thumb):
                            os.remove(unknown_thumb)
                            print(f"Xoá thumbnail unknown {unknown_thumb}")

                        path = os.path.join(KNOWN_DIR, f"{name}_ID{track_id}.jpg")
                        save_queue.put((face_crop, path))
                        trackid_saved_known.add(track_id)
                        print(f"Lưu ảnh khuôn mặt {name}_ID{track_id}.jpg")

                    current_label = recording_label.get(track_id, "")
                    new_label     = f"known:{name}"
                    if current_label != new_label:
                        if current_label == "unknown":
                            video_write_queue.put((track_id, None, f"__promote__:{name}"))
                            print(f"[Process] Promote video ID{track_id} → {name}")
                        recording_label[track_id] = new_label


                else:
                    if track_id not in trackid_saved_known and track_id not in trackid_saved_unknown:
                        path = os.path.join(UNKNOWN_DIR, f"Unknown_ID{track_id}.jpg")
                        save_queue.put((face_crop, path))
                        trackid_saved_unknown.add(track_id)
                        print(f"Lưu ảnh khuôn mặt Unknown_ID{track_id}.jpg")

                    if track_id not in recording_label:
                        recording_label[track_id] = "unknown"

            for x1, y1, x2, y2, track_id, _ in tracks:
                lbl = recording_label.get(track_id)
                if lbl is not None:
                    video_write_queue.put((track_id, annotated.copy(), lbl))

            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass

            result_queue.put(annotated)

    except Exception as e:
        print(f"[process_frames] Lỗi: {e}")
    finally:
        print("process_frames dừng")


# ===================== DISPLAY FRAMES =====================
def display_frames():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while running.is_set():
            try:
                frame = result_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                running.clear()
                break

    except Exception as e:
        print(f"[display_frames] Lỗi: {e}")
    finally:
        cv2.destroyAllWindows()
        print("display_frames dừng")


threads = [
    threading.Thread(target=capture_frames,  daemon=True, name="T1-Capture"),
    threading.Thread(target=process_frames,  daemon=True, name="T2-Process"),
    threading.Thread(target=display_frames,  daemon=True, name="T3-Display"),
]

for t in threads:
    t.start()

print("🚀 Đang chạy Face Recognition (nhấn 'q' để thoát)")
print(f"📹 Video known   lưu tại: {KNOWN_VIDEO_DIR}")
print(f"📹 Video unknown lưu tại: {UNKNOWN_VIDEO_DIR}")

try:
    while running.is_set():
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n⏹  Đang dừng...")
finally:
    running.clear()
    for t in threads:
        t.join(timeout=3.0)

    video_write_queue.put(("__STOP__", None, "__stop__"))
    video_write_queue.join()

    for q in (frame_queue, result_queue):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    cap.release()
    print("✅ Đã dừng hoàn toàn")