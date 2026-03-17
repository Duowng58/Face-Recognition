"""
Attendance App – Tkinter version (no PySide6).

Full GUI application using tkinter + PIL for display,
replicating all features of the PySide6 AttendanceWindow.

Usage:
    python app_tkinter/main.py
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Optional, Tuple

import cv2
import numpy as np
from bson import ObjectId
from PIL import Image, ImageTk

# ── project root ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# from app.config import VIDEO
from app.config import (
    ANNOY_INDEX_PATH,
    AVATAR_DIR,
    CHECKIN_DIR,
    DEFAULT_RTMP_URL,
    DEFAULT_RTSP_URL,
    EMBEDDING_DIM,
    FACE_DATA_DIR,
    FONT_PATH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAPPING_PATH,
    MIN_BBOX_AREA,
    SIM_THRESHOLD,
    TREE,
    VIDEO_FPS,
    VIDEO,
)
from app.services.recognition import RecognitionService
from app.services.streaming import StreamingService
from app.utils.cv2_helper import check_blur_laplacian, cv2_putText_utf8
from app.utils.image_utils import (
    get_attendance_frame_path,
    get_student_avatar_path,
)
from app.utils.mongodb_access import (
    Attendance,
    AttendanceRepository,
    MongoClientSingleton,
    Student,
    StudentRepository,
    now_local,
    start_of_today_local,
    to_local,
)
from app_tkinter.face_tracker import Tracker


# ══════════════════════════════════════════════════════════════
# Dark-theme colours
# ══════════════════════════════════════════════════════════════

BG = "#1f1f1f"
BG2 = "#2a2a2a"
FG = "#e5e7eb"
ACCENT = "#3b82f6"
ACCENT_HOVER = "#2563eb"
BORDER = "#3d3d3d"
GREEN = "#22c55e"
RED = "#ef4444"


def _configure_dark_style():
    """Apply a dark colour scheme to ttk widgets."""
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(".", background=BG, foreground=FG, fieldbackground=BG2, bordercolor=BORDER)
    style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 11))
    style.configure("TButton", background=BG2, foreground=FG, font=("Segoe UI", 10), padding=6)
    style.map("TButton", background=[("active", ACCENT)])
    style.configure("Accent.TButton", background=ACCENT, foreground="white", font=("Segoe UI", 10, "bold"))
    style.map("Accent.TButton", background=[("active", ACCENT_HOVER)])
    style.configure("TEntry", fieldbackground=BG2, foreground=FG, insertcolor=FG)
    style.configure("TCombobox", fieldbackground=BG2, foreground=FG, selectbackground=ACCENT)
    style.configure("TCheckbutton", background=BG, foreground=FG)
    style.configure("TLabelframe", background=BG, foreground=FG)
    style.configure("TLabelframe.Label", background=BG, foreground=FG, font=("Segoe UI", 11, "bold"))
    style.configure("Treeview", background=BG2, foreground=FG, fieldbackground=BG2,
                     rowheight=28, font=("Segoe UI", 10))
    style.configure("Treeview.Heading", background=BG, foreground=FG, font=("Segoe UI", 10, "bold"))
    style.map("Treeview", background=[("selected", ACCENT)])
    style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), foreground="#d9e2ef", background=BG)
    style.configure("Section.TLabel", font=("Segoe UI", 12, "bold"), foreground=FG, background=BG)


# ══════════════════════════════════════════════════════════════
# Helper: cv2 frame → PIL ImageTk.PhotoImage
# ══════════════════════════════════════════════════════════════

def cv2_to_photo(frame: np.ndarray, size: Tuple[int, int] | None = None) -> ImageTk.PhotoImage:
    """Convert a BGR cv2 frame to a Tkinter-compatible PhotoImage."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    if size:
        img = img.resize(size, Image.LANCZOS)
    return ImageTk.PhotoImage(img)


def load_image_file(path: str, size: Tuple[int, int] = (120, 120)) -> Optional[ImageTk.PhotoImage]:
    """Load an image file and return a PhotoImage, or None."""
    if not path or not os.path.exists(path):
        return None
    try:
        img = Image.open(path)
        img.thumbnail(size, Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
# Student Picker Dialog
# ══════════════════════════════════════════════════════════════

class StudentPickerDialog(tk.Toplevel):
    def __init__(self, parent, students: list[Student], classrooms: list[dict], avatar_dir: str):
        super().__init__(parent)
        self.title("Chọn học sinh")
        self.configure(bg=BG)
        self.geometry("600x400")
        self.transient(parent)
        self.grab_set()

        self._students = students
        self._class_map = {str(c.get("_id")): c.get("name", "") for c in classrooms}
        self._avatar_dir = avatar_dir
        self.result: Optional[Student] = None

        # Tree
        cols = ("id", "class", "name")
        self._tree = ttk.Treeview(self, columns=cols, show="headings", height=15)
        self._tree.heading("id", text="ID")
        self._tree.heading("class", text="Lớp")
        self._tree.heading("name", text="Tên")
        self._tree.column("id", width=200)
        self._tree.column("class", width=120)
        self._tree.column("name", width=200)
        self._tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        for i, s in enumerate(students):
            cls_name = self._class_map.get(str(s.class_id), "")
            self._tree.insert("", tk.END, iid=str(i), values=(str(s.id), cls_name, s.name))

        self._tree.bind("<Double-1>", lambda _: self._on_ok())

        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(btn_frame, text="Chọn", style="Accent.TButton", command=self._on_ok).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Hủy", command=self.destroy).pack(side=tk.RIGHT, padx=4)

        # center on screen after layout is done
        try:
            self.update_idletasks()
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()
            w = self.winfo_width()
            h = self.winfo_height()
            x = max((sw - w) // 2, 0)
            y = max((sh - h) // 2, 0)
            self.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _on_ok(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        self.result = self._students[idx]
        self.destroy()

    @staticmethod
    def pick(parent, students, classrooms, avatar_dir) -> Optional[Student]:
        dlg = StudentPickerDialog(parent, students, classrooms, avatar_dir)
        dlg.wait_window()
        return dlg.result


# ══════════════════════════════════════════════════════════════
# Update Student Dialog
# ══════════════════════════════════════════════════════════════

class UpdateStudentDialog(tk.Toplevel):
    def __init__(self, parent, attendance: Attendance, student: Optional[Student],
                 student_id: Optional[ObjectId], classrooms: list[dict],
                 avatar_dir: str, checkin_dir: str):
        super().__init__(parent)
        self.title("Cập nhật học sinh")
        self.configure(bg=BG)
        self.geometry("540x620")
        self.transient(parent)
        self.grab_set()

        self._attendance = attendance
        self._classrooms = classrooms
        self._avatar_dir = avatar_dir
        self._checkin_dir = checkin_dir
        self._student_id = student_id
        self._selected_student: Optional[Student] = None

        student_repo = StudentRepository()
        self._all_students = student_repo.find()

        self.result = None  # will be set on accept

        # keep references to PhotoImages so they don't get garbage collected
        self._photo_refs: list = []

        # ── images row ─────────────────────────
        img_frame = tk.Frame(self, bg=BG)
        img_frame.pack(fill=tk.X, padx=12, pady=8)

        # student avatar
        sa_col = tk.Frame(img_frame, bg=BG)
        sa_col.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(sa_col, text="Ảnh học sinh", bg=BG, fg=FG, font=("Segoe UI", 10)).pack()
        sa_box = tk.Frame(sa_col, bg=BG2, width=180, height=180, relief="groove", bd=1)
        sa_box.pack(pady=4)
        sa_box.pack_propagate(False)
        self._student_avatar_label = tk.Label(sa_box, bg=BG2)
        self._student_avatar_label.pack(expand=True)

        # attendance avatar
        aa_col = tk.Frame(img_frame, bg=BG)
        aa_col.pack(side=tk.LEFT)
        tk.Label(aa_col, text="Ảnh điểm danh", bg=BG, fg=FG, font=("Segoe UI", 10)).pack()
        aa_box = tk.Frame(aa_col, bg=BG2, width=180, height=180, relief="groove", bd=1)
        aa_box.pack(pady=4)
        aa_box.pack_propagate(False)
        self._attendance_avatar_label = tk.Label(aa_box, bg=BG2)
        self._attendance_avatar_label.pack(expand=True)

        self._refresh_student_avatar(student_id)
        self._refresh_attendance_avatar(attendance.id)

        # ── form ───────────────────────────────
        form = tk.Frame(self, bg=BG)
        form.pack(fill=tk.X, padx=12, pady=4)

        # ID + picker
        tk.Label(form, text="ID:", bg=BG, fg=FG, font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", pady=4)
        id_row = tk.Frame(form, bg=BG)
        id_row.grid(row=0, column=1, sticky="ew", pady=4)
        self._id_var = tk.StringVar(value=str(student.id) if student else "--")
        id_entry = ttk.Entry(id_row, textvariable=self._id_var, state="readonly", width=30)
        id_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(id_row, text="Chọn", style="Accent.TButton", command=self._handle_student_pick).pack(side=tk.LEFT, padx=(6, 0))

        # Name
        tk.Label(form, text="Tên:", bg=BG, fg=FG, font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", pady=4)
        self._name_var = tk.StringVar(value=student.name if student else attendance.student_name)
        ttk.Entry(form, textvariable=self._name_var, width=36).grid(row=1, column=1, sticky="ew", pady=4)

        # Class combo
        tk.Label(form, text="Lớp:", bg=BG, fg=FG, font=("Segoe UI", 10)).grid(row=2, column=0, sticky="w", pady=4)
        class_frame = tk.Frame(form, bg=BG)
        class_frame.grid(row=2, column=1, sticky="ew", pady=4)

        self._class_names = [c.get("name", "") for c in classrooms]
        self._class_ids = [c.get("_id") for c in classrooms]
        display_values = self._class_names + ["+ Thêm lớp mới"]

        self._class_var = tk.StringVar()
        self._class_combo = ttk.Combobox(class_frame, textvariable=self._class_var,
                                          values=display_values, state="readonly", width=33)
        self._class_combo.pack(fill=tk.X)
        self._class_combo.bind("<<ComboboxSelected>>", self._handle_class_change)

        self._new_class_var = tk.StringVar()
        self._new_class_entry = ttk.Entry(class_frame, textvariable=self._new_class_var, width=36)
        # hidden by default

        # Pre-select class
        if student and student.class_id is not None:
            self._set_class_selection(student.class_id)

        form.columnconfigure(1, weight=1)

        # ── buttons ────────────────────────────
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(fill=tk.X, padx=12, pady=12)
        ttk.Button(btn_frame, text="Lưu", style="Accent.TButton", command=self._on_accept).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Hủy", command=self.destroy).pack(side=tk.RIGHT, padx=4)

        # center on screen after fully constructed so it appears in middle of display
        try:
            self.update_idletasks()
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()
            w = self.winfo_width()
            h = self.winfo_height()
            x = max((sw - w) // 2, 0)
            y = max((sh - h) // 2, 0)
            self.geometry(f"+{x}+{y}")
        except Exception:
            pass

    # ── helpers ────────────────────────────────

    def _set_photo(self, label: tk.Label, path: str, size=(180, 180)):
        photo = load_image_file(path, size)
        if photo:
            self._photo_refs.append(photo)
            label.configure(image=photo)
        else:
            label.configure(image="", text="Không có ảnh")

    def _refresh_student_avatar(self, student_id):
        path = get_student_avatar_path(self._avatar_dir, student_id)
        self._set_photo(self._student_avatar_label, path if path else "")

    def _refresh_attendance_avatar(self, attendance_id):
        path = get_attendance_frame_path(self._checkin_dir, self._attendance.time, attendance_id)
        self._attendance_image_path = path
        self._set_photo(self._attendance_avatar_label, path if path else "")

    def _set_class_selection(self, class_id):
        for i, cid in enumerate(self._class_ids):
            if str(cid) == str(class_id):
                self._class_combo.current(i)
                return

    def _handle_student_pick(self):
        picked = StudentPickerDialog.pick(self, self._all_students, self._classrooms, self._avatar_dir)
        if picked is None:
            return
        self._selected_student = picked
        self._id_var.set(str(picked.id))
        self._name_var.set(picked.name)
        self._set_class_selection(picked.class_id)
        self._refresh_student_avatar(picked.id)

    def _handle_class_change(self, _event=None):
        idx = self._class_combo.current()
        if idx == len(self._class_names):  # "+ Thêm lớp mới"
            self._new_class_entry.pack(fill=tk.X, pady=(4, 0))
        else:
            self._new_class_entry.pack_forget()

    def _on_accept(self):
        new_name = self._name_var.get().strip()
        if not new_name:
            messagebox.showwarning("Thiếu dữ liệu", "Vui lòng nhập tên học sinh.", parent=self)
            return

        idx = self._class_combo.current()
        if idx < 0:
            messagebox.showwarning("Thiếu dữ liệu", "Vui lòng chọn lớp.", parent=self)
            return

        if idx == len(self._class_names):
            new_class_name = self._new_class_var.get().strip()
            if not new_class_name:
                messagebox.showwarning("Thiếu dữ liệu", "Vui lòng nhập tên lớp mới.", parent=self)
                return
            client = MongoClientSingleton.get_client()
            ins = client.db["classrooms"].insert_one({"name": new_class_name})
            new_class_id = ins.inserted_id
            class_name = new_class_name
        else:
            new_class_id = self._class_ids[idx]
            class_name = self._class_names[idx]

        selected_student_id = self._selected_student.id if self._selected_student else None

        # Save attendance avatar image to disk if needed
        att_avatar_path = self._attendance_image_path if hasattr(self, "_attendance_image_path") else None

        self.result = {
            "new_name": new_name,
            "new_class_id": new_class_id,
            "class_name": class_name,
            "selected_student_id": selected_student_id,
            "attendance_avatar_path": att_avatar_path,
        }
        self.destroy()

    @staticmethod
    def show(parent, attendance, student, student_id, classrooms, avatar_dir, checkin_dir):
        # Preserve parent's fullscreen state: on some platforms showing a modal
        # Toplevel can toggle the parent's fullscreen. Capture and restore it.
        try:
            was_fullscreen = bool(parent.attributes("-fullscreen"))
        except Exception:
            was_fullscreen = False

        dlg = UpdateStudentDialog(parent, attendance, student, student_id,
                                  classrooms, avatar_dir, checkin_dir)
        dlg.wait_window()

        # restore fullscreen if it was previously enabled
        try:
            if was_fullscreen:
                parent.attributes("-fullscreen", True)
        except Exception:
            pass

        return dlg.result


# ══════════════════════════════════════════════════════════════
# Main Application Window
# ══════════════════════════════════════════════════════════════

class AttendanceApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Hệ thống điểm danh khuôn mặt")
        self.root.configure(bg=BG)
        # Start the app fullscreen. Bind F11 to toggle and Esc to exit.
        # try:
        #     self.root.attributes("-fullscreen", True)
        #     self._is_fullscreen = True
        # except Exception:
        #     # Fallback to a reasonable window size if fullscreen not supported
        #     self.root.geometry("1280x780")
        #     self._is_fullscreen = False
        self.root.minsize(1180, 720)
        self.root.state('zoomed')
        # self.root.attributes("")
        # key bindings for convenience
        # self.root.bind("<F11>", lambda e: self._toggle_fullscreen())
        # self.root.bind("<Escape>", lambda e: self._exit_fullscreen())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        _configure_dark_style()

        # ── state ──────────────────────────────
        self._running = False
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._detect_thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_faces: list = []
        self._last_seen: dict[str, float] = {}
        self._trackid_to_name: dict = {}
        self._attendance_selected: Optional[Attendance] = None
        self._attendance_selected_iid: Optional[str] = None
        self.list_classrooms: list = []
        self._photo_refs: list = []  # prevent GC of PhotoImages
        self._Frame_FPS: float = 25

        # save queue
        self._save_queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._save_worker, daemon=True).start()

        # ── services ───────────────────────────
        self._recognition = RecognitionService(
            face_data_dir=FACE_DATA_DIR,
            annoy_index_path=ANNOY_INDEX_PATH,
            mapping_path=MAPPING_PATH,
            embedding_dim=EMBEDDING_DIM,
            tree=TREE,
            sim_threshold=SIM_THRESHOLD,
        )
        self._streaming = StreamingService(
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            rtmp_url=DEFAULT_RTMP_URL,
        )
        self._tracker = Tracker()
        self._tracker.on_disappeared = self._handle_disappeared

        os.makedirs(CHECKIN_DIR, exist_ok=True)

        # ── build UI ──────────────────────────
        self._build_ui()
        self._load_recognition_assets()
        self.load_params()
        # Auto-start capture on launch
        try:
            self.root.after(100, self._start_capture)
        except Exception:
            pass

    # ==================================================================
    # UI Construction
    # ==================================================================

    def _build_ui(self) -> None:
        # Title
        ttk.Label(self.root, text="Hệ thống điểm danh khuôn mặt", style="Title.TLabel"
                  ).pack(pady=(12, 6))

        # ── toolbar row ───────────────────────
        toolbar = tk.Frame(self.root, bg=BG)
        toolbar.pack(fill=tk.X, padx=16, pady=(0, 8))

        # tk.Label(toolbar, text="Nguồn video:", bg=BG, fg=FG, font=("Segoe UI", 10)).pack(side=tk.LEFT)
        self._source_var = tk.StringVar(value="Video")
        source_combo = ttk.Combobox(toolbar, textvariable=self._source_var,
                                     values=["Webcam", "RTSP", "Video"], state="readonly", width=10)
        # source_combo.pack(side=tk.LEFT, padx=(4, 12))

        tk.Label(toolbar, text="RTSP URL:", bg=BG, fg=FG, font=("Segoe UI", 10)).pack(side=tk.LEFT)
        self._rtsp_var = tk.StringVar(value=DEFAULT_RTSP_URL)
        ttk.Entry(toolbar, textvariable=self._rtsp_var, width=50).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        self._stream_var = tk.BooleanVar(value=False)
        # ttk.Checkbutton(toolbar, text="Livestream", variable=self._stream_var,
        #                  command=self._toggle_streaming).pack(side=tk.LEFT, padx=(12, 0))

        # ── content: left (video) + right (info) ──
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        self._build_video_panel(content)
        self._build_info_panel(content)

    def _build_video_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="")
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        # Video canvas
        self._video_canvas = tk.Canvas(frame, bg="black", highlightthickness=0,
                                        width=640, height=480)
        self._video_canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self._video_image_id = None

        # Status label
        self._status_var = tk.StringVar(value="Thông báo: Chưa có dữ liệu điểm danh.")
        status_bar = tk.Frame(frame, bg=BG2, padx=12, pady=6)
        status_bar.grid(row=1, column=0, sticky="ew", padx=8)
        tk.Label(status_bar, textvariable=self._status_var, bg=BG2, fg=FG,
                 font=("Segoe UI", 10), anchor="w").pack(fill=tk.X)

        # Buttons
        btn_row = tk.Frame(frame, bg=BG)
        btn_row.grid(row=2, column=0, sticky="ew", padx=8, pady=8)
        self._start_btn = ttk.Button(btn_row, text="Bắt đầu", style="Accent.TButton",
                                      command=self._toggle_capture)
        self._start_btn.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row, text="Nhập dữ liệu học sinh", command=self._import_data).pack(side=tk.LEFT, padx=6)
        # ttk.Button(btn_row, text="Check", command=self._check_things).pack(side=tk.LEFT, padx=6)

    def _build_info_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="")
        frame.grid(row=0, column=1, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Thông tin điểm danh", style="Section.TLabel").pack(anchor="w", padx=8, pady=(8, 4))

        # ── avatar row ─────────────────────────
        avatar_row = tk.Frame(frame, bg=BG)
        avatar_row.pack(fill=tk.X, padx=8, pady=4)

        # Use pixel-sized frames as containers (tk.Label width/height are in chars)
        av_frame1 = tk.Frame(avatar_row, bg=BG2, width=120, height=120,
                             relief="groove", bd=1)
        av_frame1.pack(side=tk.LEFT, padx=(0, 6))
        av_frame1.pack_propagate(False)  # keep fixed pixel size
        self._avatar_label = tk.Label(av_frame1, bg=BG2)
        self._avatar_label.pack(expand=True)

        av_frame2 = tk.Frame(avatar_row, bg=BG2, width=120, height=120,
                             relief="groove", bd=1)
        av_frame2.pack(side=tk.LEFT, padx=(0, 6))
        av_frame2.pack_propagate(False)
        self._student_img_label = tk.Label(av_frame2, bg=BG2)
        self._student_img_label.pack(expand=True)

        ttk.Button(avatar_row, text="Cập nhật", command=self._open_update_student_dialog).pack(
            side=tk.LEFT, anchor="n", padx=6)

        # ── info form ──────────────────────────
        info_frame = tk.Frame(frame, bg=BG)
        info_frame.pack(fill=tk.X, padx=8, pady=4)

        labels = ["ID Học sinh:", "Tên Học sinh:", "Lớp:", "Thời gian:"]
        self._id_var_info = tk.StringVar(value="--")
        self._name_var_info = tk.StringVar(value="--")
        self._class_var_info = tk.StringVar(value="--")
        self._time_var_info = tk.StringVar(value="--")
        vars_ = [self._id_var_info, self._name_var_info, self._class_var_info, self._time_var_info]

        for i, (lbl, var) in enumerate(zip(labels, vars_)):
            tk.Label(info_frame, text=lbl, bg=BG, fg=FG, font=("Segoe UI", 10), anchor="w").grid(
                row=i, column=0, sticky="w", pady=2)
            tk.Label(info_frame, textvariable=var, bg=BG, fg=FG, font=("Segoe UI", 10, "bold"), anchor="w").grid(
                row=i, column=1, sticky="w", padx=(8, 0), pady=2)

        # ── divider ───────────────────────────
        ttk.Separator(frame, orient="horizontal").pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(frame, text="Lịch sử điểm danh", style="Section.TLabel").pack(anchor="w", padx=8)

        # ── history treeview ──────────────────
        tree_frame = tk.Frame(frame, bg=BG)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        cols = ("class", "name", "time", "score")
        self._history_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=10)
        self._history_tree.heading("class", text="Lớp")
        self._history_tree.heading("name", text="Tên học sinh")
        self._history_tree.heading("time", text="Thời gian")
        self._history_tree.heading("score", text="Score")
        self._history_tree.column("class", width=80, minwidth=60)
        self._history_tree.column("name", width=140, minwidth=100, stretch=True)
        self._history_tree.column("time", width=80, minwidth=60)
        self._history_tree.column("score", width=60, minwidth=50)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self._history_tree.yview)
        self._history_tree.configure(yscrollcommand=scrollbar.set)
        self._history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._history_tree.bind("<<TreeviewSelect>>", self._on_history_select)

        # map iid → Attendance object
        self._history_data: dict[str, Attendance] = {}

    # ==================================================================
    # Recognition helpers
    # ==================================================================

    def _load_recognition_assets(self) -> None:
        self._status_var.set("Thông báo: Đang tải mô hình nhận diện...")
        self._recognition.load_assets(
            lambda msg: self.root.after(0, lambda: self._status_var.set(msg))
        )

    def _build_face(self) -> None:
        self._recognition.build_face(
            on_complete=lambda msg: self.root.after(0, lambda: self._status_var.set(msg))
        )

    def _recognize(self, embedding: np.ndarray) -> Tuple[str, float]:
        return self._recognition.recognize(embedding)

    # ==================================================================
    # Save worker
    # ==================================================================

    def _save_worker(self) -> None:
        while True:
            frame, path = self._save_queue.get()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, frame)
            self._save_queue.task_done()

    # ==================================================================
    # Tracker disappeared
    # ==================================================================

    def _handle_disappeared(self, track_id: int) -> None:
        print(f"Object {track_id} disappeared")
        tracker = self._trackid_to_name.get(track_id)
        if tracker is None:
            return

        tracker_snapshot = {
            "frames": list(tracker.get("frames", [])),
            "frame": tracker.get("frame"),
            "name": tracker.get("name"),
            "score": tracker.get("score", 0),
            "attendance_id": tracker.get("attendance_id"),
            "video_writer": tracker.get("video_writers"),
        }
        self._trackid_to_name.pop(track_id, None)

        def _process():
            timestamp = now_local()

            def save_frames(attendance_id):
                now_path = os.path.join(CHECKIN_DIR, timestamp.strftime("%Y-%m-%d"), str(attendance_id))
                if tracker_snapshot["frame"] is not None:
                    self._save_queue.put((tracker_snapshot["frame"], os.path.join(now_path, "frame.jpg")))
                for idx, (_, __, frm) in enumerate(tracker_snapshot["frames"]):
                    self._save_queue.put((frm, os.path.join(now_path, "frames", f"frame_{idx}.jpg")))
                vw = tracker_snapshot.get("video_writer")
                if vw is not None:
                    vw.release()
                    old = os.path.join(CHECKIN_DIR, f"tmp_video_{track_id}.mp4")
                    new = os.path.join(now_path, "video.mp4")
                    os.makedirs(os.path.dirname(new), exist_ok=True)
                    if os.path.exists(old):
                        os.rename(old, new)

            if tracker_snapshot.get("name") == "Unknown":
                if tracker_snapshot.get("frame") is None or len(tracker_snapshot["frames"]) < 10:
                    return
                att_repo = AttendanceRepository()
                att_id = ObjectId()
                save_frames(att_id)
                att = Attendance(id=att_id, time=timestamp, student_name="Unknown",
                                 score=tracker_snapshot.get("score", 0))
                att_repo.insert(att)
                self._build_unknown_face(tracker_snapshot, att)
                self.root.after(0, lambda: self._append_history_row(att))
            else:
                att_id = tracker_snapshot.get("attendance_id")
                if att_id is not None:
                    save_frames(att_id)
                else:
                    vw = tracker_snapshot.get("video_writer")
                    if vw is not None:
                        vw.release()
                        old = os.path.join(CHECKIN_DIR, f"tmp_video_{track_id}.mp4")
                        if os.path.exists(old):
                            os.remove(old)

        threading.Thread(target=_process, daemon=True).start()

    def _build_unknown_face(self, tracker: dict, att: Attendance) -> None:
        embeddings = []
        if len(tracker["frames"]) > 10:
            for _, __, face_crop in tracker["frames"]:
                faces = self._recognition.face_app.get(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                if faces:
                    embeddings.append(faces[0].embedding)
            np.save(os.path.join(FACE_DATA_DIR, f"{att.id}.npy"), embeddings)
            print("[🔁] Rebuild Annoy Index")
            self._build_face()

    # ==================================================================
    # Data loading
    # ==================================================================

    def load_params(self) -> None:
        self._refresh_classrooms()
        att_repo = AttendanceRepository()
        records = att_repo.find({"time": {"$gte": start_of_today_local()}})

        self._clear_selected()
        for item in self._history_tree.get_children():
            self._history_tree.delete(item)
        self._history_data.clear()

        for r in records:
            self._append_history_row(r)

    def _refresh_classrooms(self) -> None:
        try:
            client = MongoClientSingleton.get_client()
            self.list_classrooms = list(client.db["classrooms"].find())
        except Exception as e:
            print(f"[WARN] Cannot load classrooms: {e}")
            self.list_classrooms = []

    def _clear_selected(self) -> None:
        self._attendance_selected = None
        self._id_var_info.set("--")
        self._name_var_info.set("--")
        self._class_var_info.set("--")
        self._time_var_info.set("--")
        self._status_var.set("-")
        self._avatar_label.configure(image="")
        self._student_img_label.configure(image="")

    # ==================================================================
    # History
    # ==================================================================

    def _append_history_row(self, record: Attendance) -> None:
        iid = str(record.id or id(record))
        self._history_tree.insert("", 0, iid=iid, values=(
            record.student_classroom,
            record.student_name,
            to_local(record.time).strftime("%H:%M:%S"),
            f"{record.score:.2f}",
        ))
        self._history_data[iid] = record

    def _on_history_select(self, _event=None) -> None:
        sel = self._history_tree.selection()
        if not sel:
            return
        iid = sel[0]
        self._attendance_selected_iid = iid
        record = self._history_data.get(iid)
        if record:
            self._attendance_selected = record
            self._update_attendance_panel(record)

    # ==================================================================
    # Update student dialog
    # ==================================================================

    def _open_update_student_dialog(self) -> None:
        if self._attendance_selected is None:
            messagebox.showinfo("Thiếu dữ liệu", "Chưa chọn học sinh để cập nhật.")
            return

        student_repo = StudentRepository()
        attendance_repo = AttendanceRepository()

        student = None
        student_id = self._attendance_selected.student_id
        if student_id is not None:
            try:
                if not isinstance(student_id, ObjectId):
                    student_id = ObjectId(str(student_id))
                student = student_repo.get(student_id)
            except Exception:
                student_id = None

        result = UpdateStudentDialog.show(
            parent=self.root,
            attendance=self._attendance_selected,
            student=student,
            student_id=student_id,
            classrooms=self.list_classrooms,
            avatar_dir=AVATAR_DIR,
            checkin_dir=CHECKIN_DIR,
        )
        if result is None:
            return

        self._refresh_classrooms()

        if result["selected_student_id"] is not None:
            student_id = result["selected_student_id"]

        if student_id is None:
            student = Student(
                id=self._attendance_selected.id,
                name=result["new_name"],
                class_id=result["new_class_id"],
            )
            student_id = student_repo.insert(student)
            os.makedirs(AVATAR_DIR, exist_ok=True)
            att_path = result.get("attendance_avatar_path")
            if att_path and os.path.exists(att_path):
                import shutil
                shutil.copy2(att_path, os.path.join(AVATAR_DIR, f"{student_id}.jpg"))
        else:
            student_repo.update(student_id, {
                "name": result["new_name"],
                "class_id": result["new_class_id"],
            })

        attendance_repo.update(self._attendance_selected.id, {
            "student_id": student_id,
            "student_name": result["new_name"],
            "student_classroom": result["class_name"],
        })

        self._name_var_info.set(result["new_name"])
        self._class_var_info.set(result["class_name"])

        if self._attendance_selected:
            self._attendance_selected.student_name = result["new_name"]
            self._attendance_selected.student_classroom = result["class_name"]
            self._attendance_selected.student_id = student_id

        self._update_student_image(student_id)

        if self._attendance_selected_iid:
            self._history_tree.item(self._attendance_selected_iid, values=(
                result["class_name"],
                result["new_name"],
                to_local(self._attendance_selected.time).strftime("%H:%M:%S"),
                f"{self._attendance_selected.score:.2f}",
            ))

    # ==================================================================
    # Import data / check
    # ==================================================================

    def _import_data(self) -> None:
        print("Importing data...")
        self.load_params()
        # handle disappear all self._trackid_to_name 
        for tid in list(self._trackid_to_name.keys()):
            self._handle_disappeared(tid)

    def _check_things(self) -> None:
        print("Checking things...")
        for tid, trk in self._trackid_to_name.items():
            print(f"Track ID: {tid}, frames: {len(trk['frames'])}, name: {trk.get('name', '--')}")

    # ==================================================================
    # Capture / Video
    # ==================================================================

    def _toggle_capture(self) -> None:
        if self._running:
            self._stop_capture()
        else:
            self._start_capture()

    def _start_capture(self) -> None:
        if self._running:
            return

        source: int | str = 0
        if self._source_var.get() == "RTSP":
            source = self._rtsp_var.get().strip()
            if not source:
                messagebox.showwarning("Thiếu RTSP", "Vui lòng nhập RTSP URL.")
                return
        elif self._source_var.get() == "Video":
            source = VIDEO

        backend = cv2.CAP_ANY if source == 0 else cv2.CAP_FFMPEG
        self._capture = cv2.VideoCapture(source, backend)
        if source != 0:
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._capture.isOpened():
            messagebox.showerror("Lỗi", "Không mở được nguồn video.")
            return

        self._Frame_FPS = self._capture.get(cv2.CAP_PROP_FPS) or 25
        self._running = True
        self._start_btn.configure(text="Kết thúc")
        self._latest_faces = []

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self._detect_thread = threading.Thread(target=self._detect_worker, daemon=True)
        self._detect_thread.start()

        self._streaming.start(self._Frame_FPS, lambda: self._running)

        # start render loop
        self._render_frame()

    def _toggle_streaming(self) -> None:
        enabled = self._stream_var.get()
        fps = self._Frame_FPS or 25
        self._streaming.toggle(enabled, fps, lambda: self._running)

    def _stop_capture(self) -> None:
        if not self._running and self._capture is None:
            return
        self._running = False
        self._start_btn.configure(text="Bắt đầu")

        for t in (self._capture_thread, self._detect_thread):
            if t and t.is_alive():
                t.join(timeout=1)

        self._streaming.stop()
        self._release_capture()

    def _release_capture(self) -> None:
        if self._capture is None:
            return
        try:
            if self._capture.isOpened():
                self._capture.release()
        except cv2.error:
            pass
        finally:
            self._capture = None
        try:
            self._tracker.release()
        except Exception:
            pass

    def _capture_loop(self) -> None:
        is_webcam = self._source_var.get() == "Webcam"
        while self._running and self._capture is not None:
            try:
                ret, frame = self._capture.read()
            except cv2.error:
                break
            if not ret:
                time.sleep(0.01)
                continue
            if is_webcam:
                frame = cv2.flip(frame, 1)
            # frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            with self._frame_lock:
                self._latest_frame = frame.copy()
            time.sleep(1 / (self._Frame_FPS or 25))
        self._running = False

    # ==================================================================
    # Detection worker
    # ==================================================================

    def _detect_worker(self) -> None:
        trackid_saved_known: set = set()
        detect_frame_count = 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        while self._running:
            # Wait for a frame to be captured
            if self._latest_frame is None:
                time.sleep(0.01)
                continue

            # Ensure recognition model is loaded before attempting detection
            if not getattr(self._recognition, "face_app", None):
                # recognition not ready yet, wait a bit
                time.sleep(0.1)
                continue

            detect_frame_count += 1
            with self._frame_lock:
                frame_copy = self._latest_frame.copy()

            rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            faces = self._recognition.face_app.get(rgb)

            rects, embeddings = [], []
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                if (x2 - x1) * (y2 - y1) < MIN_BBOX_AREA:
                    continue
                rects.append([x1, y1, x2, y2])
                embeddings.append(face.embedding)

            tracks = self._tracker.update(rects, classId="face")
            frame = frame_copy.copy()
            raw_frame = frame_copy.copy()
            rects2: list = []

            for track in tracks:
                x1, y1, x2, y2, track_id, _ = track

                matched_embedding = None
                for rect, emb in zip(rects, embeddings):
                    if abs(x1 - rect[0]) < 15 and abs(y1 - rect[1]) < 15:
                        matched_embedding = emb
                        break
                if matched_embedding is None:
                    continue

                if self._trackid_to_name.get(track_id) is None:
                    h, w = frame.shape[:2]
                    self._trackid_to_name[track_id] = {
                        "name": "Unknown", "score": 0.0, "student": None,
                        "frames": [], "frame": None, "attendance_id": None,
                        "video_writers": cv2.VideoWriter(
                            os.path.join(CHECKIN_DIR, f"tmp_video_{track_id}.mp4"),
                            fourcc, VIDEO_FPS, (w, h)),
                    }
                trk = self._trackid_to_name[track_id]
                name, score = self._recognize(matched_embedding)

                if trk["name"] == "Unknown":
                    trk["name"] = name
                else:
                    if name == "Unknown":
                        name = trk["name"]
                    elif name != trk["name"]:
                        continue

                trk["score"] = score
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"ID:{track_id} {name} {score:.2f}" if name != "Unknown" else f"ID:{track_id} Unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                frame_tracker = frame_copy.copy()
                cv2.rectangle(frame_tracker, (x1, y1), (x2, y2), color, 2)
                trk["video_writers"].write(frame_tracker)

                if name != "Unknown":
                    if track_id not in trackid_saved_known:
                        trackid_saved_known.add(track_id)
                        s_repo = StudentRepository()
                        found = s_repo.find({"_id": ObjectId(name)})
                        if found:
                            student = found[0]
                            trk["student"] = student
                            label = f"ID:{track_id} {student.name} {score:.2f}"
                            a_repo = AttendanceRepository()
                            existing = a_repo.find({"student_id": student.id, "time": {"$gte": start_of_today_local()}})
                            if not existing:
                                trk["attendance_id"] = self._register_attendance(student, score, raw_frame, (x1, y1, x2, y2))
                    else:
                        student = trk.get("student")
                        if student:
                            label = f"ID:{track_id} {student.name} {score:.2f}"

                if os.path.exists(FONT_PATH):
                    frame = cv2_putText_utf8(frame, label, (x1, y1 - 40), FONT_PATH, 30, color)
                else:
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # face crops
                face_crop_avatar = frame_copy[
                    max(y1 - int(abs(y1 - y2) * 0.2), 0):max(y2 + int(abs(y1 - y2) * 0.2), 0),
                    max(x1 - int(abs(x1 - x2) * 0.2), 0):max(x2 + int(abs(x1 - x2) * 0.2), 0),
                ]
                face_crop_build = frame_copy[y1:y2, x1:x2]
                if face_crop_build is None or face_crop_build.size == 0:
                    continue
                is_blur, variance = check_blur_laplacian(face_crop_build)
                is_blur = False  # disabled as in original
                if not is_blur:
                    if not any(detect_frame_count is c for _, c, _ in trk["frames"]):
                        trk["frames"].append((variance, detect_frame_count, face_crop_build.copy()))
                    if score > trk.get("score", 0):
                        trk["score"] = score
                        trk["frame"] = face_crop_avatar.copy()
                    elif variance > trk.get("variance", 0):
                        trk["frame"] = face_crop_avatar.copy()
                        trk["variance"] = variance

                trk["frames"] = sorted(trk["frames"], key=lambda x: x[0], reverse=True)[:15]
                rects2.append((track_id, color, label, [x1, y1, x2, y2]))

            self._latest_faces = rects2
            time.sleep(0.1)

    # ==================================================================
    # Attendance
    # ==================================================================

    def _register_attendance(self, student: Student, score: float,
                             frame: np.ndarray, bbox) -> Optional[ObjectId]:
        now = time.time()
        last = self._last_seen.get(student.id, 0)
        if now - last < 10:
            return None
        self._last_seen[student.id] = now
        timestamp = now_local()

        y1 = max(bbox[1] - int(abs(bbox[1] - bbox[3]) * 0.2), 0)
        y2 = max(bbox[3] + int(abs(bbox[1] - bbox[3]) * 0.2), 0)
        x1 = max(bbox[0] - int(abs(bbox[0] - bbox[2]) * 0.2), 0)
        x2 = max(bbox[2] + int(abs(bbox[0] - bbox[2]) * 0.2), 0)
        face_crop = frame[y1:y2, x1:x2]

        find_class = next((c for c in self.list_classrooms if c["_id"] == student.class_id), None)
        record = Attendance(
            student_id=student.id,
            student_name=student.name,
            student_classroom=find_class["name"] if find_class else "",
            time=timestamp,
            score=score,
        )
        repo = AttendanceRepository()
        record.id = repo.insert(record)

        # schedule UI updates on main thread
        self.root.after(0, lambda: self._update_attendance_panel(record, face_crop))
        self.root.after(0, lambda: self._append_history_row(record))
        return record.id

    def _update_attendance_panel(self, record: Attendance, face_crop: np.ndarray = None) -> None:
        self._id_var_info.set(str(record.student_id))
        self._name_var_info.set(record.student_name)
        self._class_var_info.set(record.student_classroom)
        self._time_var_info.set(to_local(record.time).strftime("%H:%M:%S"))
        self._status_var.set(
            f"Thông báo: {record.student_name} đã điểm danh lúc {to_local(record.time).strftime('%H:%M:%S')}."
        )

        if face_crop is None:
            path = get_attendance_frame_path(CHECKIN_DIR, record.time, record.id)
            if path and os.path.exists(path):
                face_crop = cv2.imread(os.path.normpath(path))

        if face_crop is not None and face_crop.size > 0:
            photo = cv2_to_photo(face_crop, (120, 120))
            self._photo_refs.append(photo)
            self._avatar_label.configure(image=photo)
        else:
            self._avatar_label.configure(image="")

        self._update_student_image(record.student_id)

    def _update_student_image(self, student_id) -> None:
        if student_id is None:
            self._student_img_label.configure(image="")
            return
        path = get_student_avatar_path(AVATAR_DIR, student_id)
        photo = load_image_file(path, (120, 120)) if path else None
        if photo:
            self._photo_refs.append(photo)
            self._student_img_label.configure(image=photo)
        else:
            self._student_img_label.configure(image="")

    # ==================================================================
    # Render (tkinter main-thread loop via .after)
    # ==================================================================

    def _render_frame(self) -> None:
        if not self._running:
            return

        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()

        if frame is not None:
            # draw rects
            for track_id, color, label, (x1, y1, x2, y2) in self._latest_faces:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if os.path.exists(FONT_PATH):
                    frame = cv2_putText_utf8(frame, label, (x1, y1 - 40), FONT_PATH, 30, color)
                else:
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # fit to canvas
            cw = self._video_canvas.winfo_width()
            ch = self._video_canvas.winfo_height()
            if cw > 1 and ch > 1:
                fh, fw = frame.shape[:2]
                scale = min(cw / fw, ch / fh)
                new_w, new_h = int(fw * scale), int(fh * scale)
                display = cv2.resize(frame, (new_w, new_h))
            else:
                display = frame

            photo = cv2_to_photo(display)
            self._photo_refs.append(photo)
            # keep only last 3 refs to avoid memory leak
            if len(self._photo_refs) > 5:
                self._photo_refs = self._photo_refs[-3:]

            self._video_canvas.delete("all")
            img_w, img_h = photo.width(), photo.height()
            cx = max((cw - img_w) // 2, 0)
            cy = max((ch - img_h) // 2, 0)
            self._video_canvas.create_image(cx, cy, anchor=tk.NW, image=photo)

            self._streaming.enqueue(frame.copy())

        # schedule next render (~33ms ≈ 30fps)
        self.root.after(33, self._render_frame)

    # ==================================================================
    # Close
    # ==================================================================

    def _on_close(self) -> None:
        print("app closing")
        self._stop_capture()
        self.root.destroy()

    # Fullscreen helpers
    def _toggle_fullscreen(self) -> None:
        try:
            self._is_fullscreen = not getattr(self, "_is_fullscreen", False)
            self.root.attributes("-fullscreen", self._is_fullscreen)
        except Exception:
            # ignore if not supported on platform
            pass

    def _exit_fullscreen(self) -> None:
        try:
            if getattr(self, "_is_fullscreen", False):
                self._is_fullscreen = False
                self.root.attributes("-fullscreen", False)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

def main() -> None:
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
