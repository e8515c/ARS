#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aspect Ratio Changer (Python) — enhanced
GUI: Tkinter
Processing: OpenCV (opencv-contrib-python), Pillow

Added features:
- Crop mode (center crop to target aspect ratio) in addition to Stretch.
- Custom W:H aspect ratio UI (button -> add to combobox).
- Preview processed single image before batch saving.
- Saves 'mode' and last custom AR to settings.json.

Install dependencies (one line):
pip install pillow tkinterdnd2 opencv-python opencv-contrib-python
"""
__author__ = "e8515c (modified)"

import os
import json
import random
import string
import threading
from pathlib import Path
from typing import List, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

APP_NAME = "ARC_e8515c"
SETTINGS_FILE = os.path.join(os.environ.get("APPDATA", str(Path.home() / ".config")), APP_NAME, "settings.json")
SUPPORTED_EXT = {"jpg", "jpeg", "png", "webp", "gif"}

# ---------------- Aspect ratio choices (popular) ----------------
AR_CHOICES = [
    "1:1",
    "4:3",
    "3:2",
    "16:10",
    "16:9",
    "21:9",
    "32:9",
    "5:4",
    "2:1",
    "18:9",
    "9:16"
]

UPSCALE_CHOICES = ["Без апскейла", "2× (ML)", "4× (ML)", "2× (fast)", "4× (fast)"]
OPT_CHOICES = ["Простая", "Средняя", "Максимальная"]
MODE_CHOICES = ["Stretch", "Crop"]  # new: Stretch (existing behavior) or Crop (center crop to AR)


# ---------------- Settings ----------------
def ensure_settings_dir():
    p = Path(SETTINGS_FILE).parent
    p.mkdir(parents=True, exist_ok=True)


def load_settings():
    ensure_settings_dir()
    f = Path(SETTINGS_FILE)
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_settings(data: dict):
    ensure_settings_dir()
    Path(SETTINGS_FILE).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------- Utils ----------------
def ar_value(label: str) -> float:
    """
    Convert 'W:H' string to numeric aspect ratio.
    Works for ANY "W:H" pair automatically.
    """
    try:
        w, h = label.split(":")
        return float(w) / float(h)
    except Exception:
        return 1.0


def target_size(w: int, h: int, ar_label: str, keep_long: bool) -> Tuple[int, int]:
    t = ar_value(ar_label)
    if keep_long:
        if w >= h:
            out_w = w
            out_h = max(1, int(round(w / t)))
        else:
            out_h = h
            out_w = max(1, int(round(h * t)))
    else:
        if w >= h:
            out_h = h
            out_w = max(1, int(round(h * t)))
        else:
            out_w = w
            out_h = max(1, int(round(w / t)))
    return out_w, out_h


def is_supported(path: Path) -> bool:
    return path.suffix.lower().lstrip(".") in SUPPORTED_EXT


def random_name(prefix: str = "e8515c_", min_len: int = 6, max_len: int = 12) -> str:
    n = random.randint(min_len, max_len)
    return prefix + "".join(random.choices(string.ascii_letters + string.digits, k=n))


def cv2_imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def cv2_imwrite_unicode(path: Path, img, ext: str, params=None) -> bool:
    try:
        ok, buf = cv2.imencode(ext, img, params or [])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


# ---------------- I/O helpers ----------------
def read_image(path: Path):
    ext = path.suffix.lower().lstrip(".")
    if ext == "gif":
        with Image.open(str(path)) as im:
            im = im.convert("RGB")
            return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    return cv2_imread_unicode(path)


def write_image(path_in: Path, out_dir: Path, bgr, opt_level: str) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    ext_in = path_in.suffix.lower().lstrip(".")
    out_ext = ext_in if ext_in in {"jpg", "jpeg", "png", "webp", "gif"} else "png"

    q_jpg = 90 if opt_level == "Простая" else 80 if opt_level == "Средняя" else 70
    q_webp = q_jpg
    c_png = 3 if opt_level == "Простая" else 6 if opt_level == "Средняя" else 9

    for _ in range(1024):
        name = random_name()
        out_path = out_dir / f"{name}.{out_ext}"
        if not out_path.exists():
            break
    else:
        out_path = out_dir / f"{name}_x.{out_ext}"

    try:
        if out_ext in {"jpg", "jpeg"}:
            return cv2_imwrite_unicode(out_path, bgr, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, q_jpg])
        elif out_ext == "png":
            return cv2_imwrite_unicode(out_path, bgr, ".png", [cv2.IMWRITE_PNG_COMPRESSION, c_png])
        elif out_ext == "webp":
            return cv2_imwrite_unicode(out_path, bgr, ".webp", [cv2.IMWRITE_WEBP_QUALITY, q_webp])
        elif out_ext == "gif":
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(str(out_path), "GIF", optimize=True)
            return True
        else:
            return cv2_imwrite_unicode(out_path, bgr, ".png")
    except Exception:
        return False


# ---------------- Upscale ----------------
def pick_model(models_dir: Path, factor: int):
    edsr = models_dir / f"EDSR_x{factor}.pb"
    espcn = models_dir / f"ESPCN_x{factor}.pb"
    if edsr.exists():
        return "edsr", edsr
    if espcn.exists():
        return "espcn", espcn
    return "", Path()


def upscale_fast(bgr, factor: int):
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)


def upscale_if_needed(bgr, models_dir: Path, upscale_choice: str):
    if upscale_choice == "Без апскейла":
        return True, bgr, ""
    if "fast" in upscale_choice:
        f = 2 if "2" in upscale_choice else 4
        try:
            return True, upscale_fast(bgr, f), ""
        except Exception as e:
            return False, bgr, str(e)

    factor = 2 if "2" in upscale_choice else 4
    name, model = pick_model(models_dir, factor)
    if not model.exists():
        return False, bgr, f"Model x{factor} not found"

    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(model))
        sr.setModel(name, factor)
        return True, sr.upsample(bgr), ""
    except Exception as e:
        return False, bgr, str(e)


# ---------------- Crop helper ----------------
def center_crop_to_ratio(bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center-crop BGR image to match target_w:target_h ratio, return cropped image."""
    if target_w <= 0 or target_h <= 0:
        return bgr
    h, w = bgr.shape[:2]
    target_ratio = float(target_w) / float(target_h)
    cur_ratio = float(w) / float(h)
    if abs(cur_ratio - target_ratio) < 1e-6:
        return bgr.copy()
    if cur_ratio > target_ratio:
        # too wide -> crop sides
        new_w = int(round(h * target_ratio))
        left = (w - new_w) // 2
        cropped = bgr[:, left:left + new_w]
    else:
        # too tall -> crop top/bottom
        new_h = int(round(w / target_ratio))
        top = (h - new_h) // 2
        cropped = bgr[top:top + new_h, :]
    return cropped


# ---------------- Pipeline ----------------
def process_one(path_in: Path, out_dir: Path, ar_label: str, keep_long: bool,
                upscale_choice: str, opt_level: str, models_dir: Path, mode: str):
    """
    Process single file:
     - read
     - either stretch to target size (old behavior) or center-crop -> resize (new crop mode)
     - upscale if requested
     - write
    """
    try:
        img = read_image(path_in)
        if img is None:
            return False, "Read failed"

        h, w = img.shape[:2]
        tw, th = target_size(w, h, ar_label, keep_long)

        if mode == "Crop":
            # center-crop source to desired aspect ratio, then resize to (tw,th)
            cropped = center_crop_to_ratio(img, tw, th)
            processed = cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        else:
            # Stretch behavior
            processed = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LANCZOS4)

        ok, up, msg = upscale_if_needed(processed, models_dir, upscale_choice)
        if not ok:
            return False, msg

        if not write_image(path_in, out_dir, up, opt_level):
            return False, "Write failed"

        return True, ""
    except Exception as e:
        return False, str(e)


def collect_inputs(files: List[str], folder: str, recursive: bool):
    res = []
    if files:
        for f in files:
            p = Path(f)
            if p.exists() and p.is_file() and is_supported(p):
                res.append(p)
    elif folder:
        p = Path(folder)
        if p.is_dir():
            if recursive:
                for root, _, names in os.walk(p):
                    for n in names:
                        pp = Path(root) / n
                        if is_supported(pp):
                            res.append(pp)
            else:
                for n in os.listdir(p):
                    pp = p / n
                    if pp.is_file() and is_supported(pp):
                        res.append(pp)
    return res


# ---------------- GUI ----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("e8515c's Aspect Ratio Changer")
        self.geometry("1100x560")
        self.resizable(False, False)

        self.selected_files: List[str] = []
        self.settings = load_settings()

        self._build_ui()
        self._load_settings()

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        ttk.Button(top, text="Выбрать файлы", command=self.on_choose_files).pack(side="left")
        ttk.Button(top, text="Выбрать папку", command=self.on_choose_input_dir).pack(side="left", padx=6)
        self.in_entry = ttk.Entry(top)
        self.in_entry.pack(side="left", fill="x", expand=True)

        row2 = ttk.Frame(self); row2.pack(fill="x", **pad)
        ttk.Button(row2, text="Папка сохранения", command=self.on_choose_output_dir).pack(side="left")
        self.out_entry = ttk.Entry(row2)
        self.out_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))

        row3 = ttk.Frame(self); row3.pack(fill="x", **pad)
        ttk.Button(row3, text="Папка моделей (SR)", command=self.on_choose_models_dir).pack(side="left")
        self.models_entry = ttk.Entry(row3)
        self.models_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))

        # Row for AR, Mode, Upscale, Opt
        row4 = ttk.Frame(self); row4.pack(fill="x", **pad)
        ttk.Label(row4, text="Aspect Ratio:").pack(side="left")
        self.ar_var = tk.StringVar(value=AR_CHOICES[0])
        self.ar_box = ttk.Combobox(row4, textvariable=self.ar_var, values=AR_CHOICES, state="readonly", width=10)
        self.ar_box.pack(side="left", padx=(4, 6))

        ttk.Button(row4, text="Custom W:H", command=self.on_custom_ar).pack(side="left", padx=(0, 10))

        ttk.Label(row4, text="Mode:").pack(side="left")
        self.mode_var = tk.StringVar(value=MODE_CHOICES[0])
        self.mode_box = ttk.Combobox(row4, textvariable=self.mode_var, values=MODE_CHOICES, state="readonly", width=8)
        self.mode_box.pack(side="left", padx=(4, 12))

        ttk.Label(row4, text="Апскейл:").pack(side="left")
        self.up_var = tk.StringVar(value=UPSCALE_CHOICES[0])
        self.up_box = ttk.Combobox(row4, textvariable=self.up_var, values=UPSCALE_CHOICES, state="readonly", width=14)
        self.up_box.pack(side="left", padx=(4, 12))

        ttk.Label(row4, text="Оптимизация:").pack(side="left")
        self.opt_var = tk.StringVar(value=OPT_CHOICES[0])
        self.opt_box = ttk.Combobox(row4, textvariable=self.opt_var, values=OPT_CHOICES, state="readonly", width=14)
        self.opt_box.pack(side="left", padx=(4, 12))

        self.rec_var = tk.BooleanVar(value=False)
        self.keep_long_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row4, text="Рекурсивно", variable=self.rec_var).pack(side="left", padx=6)
        ttk.Checkbutton(row4, text="Длинную сторону", variable=self.keep_long_var).pack(side="left", padx=(0,6))

        # Row for buttons including Preview
        row5 = ttk.Frame(self); row5.pack(fill="x", **pad)
        self.run_btn = ttk.Button(row5, text="Старт", command=self.on_start)
        self.run_btn.pack(side="left")
        ttk.Button(row5, text="Preview", command=self.on_preview).pack(side="left", padx=(6,6))
        ttk.Button(row5, text="Clear Custom ARs", command=self.on_clear_custom_ars).pack(side="left", padx=(6,6))

        # Add preview canvas area (right side)
        bottom = ttk.Frame(self)
        bottom.pack(fill="both", expand=True, **pad)

        left_panel = ttk.Frame(bottom)
        left_panel.pack(side="left", fill="both", expand=True)

        self.pb = ttk.Progressbar(left_panel, mode="determinate")
        self.pb.pack(fill="x", **pad)

        self.status = ttk.Label(left_panel, text="")
        self.status.pack(fill="x", **pad)

        # Right preview frame
        preview_frame = ttk.LabelFrame(bottom, text="Preview (processed)")
        preview_frame.pack(side="right", fill="both", padx=6, pady=4)
        self.preview_label = ttk.Label(preview_frame, text="No preview", anchor="center")
        self.preview_label.pack(fill="both", expand=True, padx=6, pady=6)
        self.preview_img = None  # keep reference to PhotoImage

    def _load_settings(self):
        s = self.settings
        self.in_entry.insert(0, s.get("lastInput", ""))
        self.out_entry.insert(0, s.get("lastOutput", ""))
        self.models_entry.insert(0, s.get("modelsDir", "models"))
        # load AR choices including any custom saved in settings
        custom_ars = s.get("custom_ars", [])
        if custom_ars:
            for a in custom_ars:
                if a not in AR_CHOICES:
                    AR_CHOICES.append(a)
        self.ar_box.config(values=AR_CHOICES)
        self.ar_var.set(s.get("ar", AR_CHOICES[0]))
        up = s.get("up", UPSCALE_CHOICES[0])
        self.up_var.set(up if up in UPSCALE_CHOICES else UPSCALE_CHOICES[0])
        self.opt_var.set(s.get("opt", OPT_CHOICES[0]))
        self.rec_var.set(bool(s.get("rec", False)))
        self.keep_long_var.set(bool(s.get("keepLong", True)))
        self.mode_var.set(s.get("mode", MODE_CHOICES[0]))

    def _save_settings(self):
        # store custom ARs present in combobox beyond defaults
        custom = [a for a in self.ar_box['values'] if a not in [
            "1:1","4:3","3:2","16:10","16:9","21:9","32:9","5:4","2:1","18:9","9:16"
        ]]
        save_settings({
            "lastInput": self.in_entry.get().strip(),
            "lastOutput": self.out_entry.get().strip(),
            "modelsDir": self.models_entry.get().strip() or "models",
            "ar": self.ar_var.get(),
            "up": self.up_var.get(),
            "opt": self.opt_var.get(),
            "rec": bool(self.rec_var.get()),
            "keepLong": bool(self.keep_long_var.get()),
            "mode": self.mode_var.get(),
            "custom_ars": custom
        })

    def on_choose_files(self):
        files = filedialog.askopenfilenames(
            title="Выберите изображения",
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.webp;*.gif"), ("All", "*.*")]
        )
        if files:
            self.selected_files = list(files)
            self.in_entry.delete(0, tk.END)
            # show first filename or indicator
            if len(files) == 1:
                self.in_entry.insert(0, files[0])
            else:
                self.in_entry.insert(0, f"[{len(files)} files selected]")

    def on_choose_input_dir(self):
        d = filedialog.askdirectory(title="Выберите папку")
        if d:
            self.selected_files = []
            self.in_entry.delete(0, tk.END)
            self.in_entry.insert(0, d)

    def on_choose_output_dir(self):
        d = filedialog.askdirectory(title="Папка сохранения")
        if d:
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, d)

    def on_choose_models_dir(self):
        d = filedialog.askdirectory(title="Папка моделей")
        if d:
            self.models_entry.delete(0, tk.END)
            self.models_entry.insert(0, d)

    def on_custom_ar(self):
        """Ask user for custom W and H, validate and add to AR combobox."""
        txt = simpledialog.askstring("Custom AR", "Enter custom aspect ratio as W:H (e.g. 7:5):", parent=self)
        if not txt:
            return
        txt = txt.strip()
        if ":" not in txt:
            messagebox.showerror("Invalid", "Format must be W:H")
            return
        parts = txt.split(":")
        try:
            w = int(parts[0]); h = int(parts[1])
            if w <= 0 or h <= 0:
                raise ValueError()
        except Exception:
            messagebox.showerror("Invalid", "W and H must be positive integers")
            return
        label = f"{w}:{h}"
        vals = list(self.ar_box['values'])
        if label not in vals:
            vals.append(label)
            self.ar_box.config(values=vals)
        self.ar_var.set(label)
        # save settings immediately
        self._save_settings()
        messagebox.showinfo("Added", f"Custom AR {label} added and selected")

    def on_clear_custom_ars(self):
        vals = list(self.ar_box['values'])
        defaults = ["1:1","4:3","3:2","16:10","16:9","21:9","32:9","5:4","2:1","18:9","9:16"]
        new_vals = [v for v in vals if v in defaults]
        self.ar_box.config(values=new_vals)
        if self.ar_var.get() not in new_vals:
            self.ar_var.set(new_vals[0])
        self._save_settings()
        messagebox.showinfo("Cleared", "Custom aspect ratios removed from the list")

    def on_start(self):
        out_dir = self.out_entry.get().strip()
        if not out_dir:
            messagebox.showerror("Ошибка", "Укажите папку сохранения")
            return
        self._save_settings()

        inputs = collect_inputs(self.selected_files, self.in_entry.get().strip(), self.rec_var.get())
        if not inputs:
            messagebox.showinfo("Информация", "Нет входных изображений")
            return

        t = threading.Thread(target=self._run_pipeline, args=(inputs, Path(out_dir)))
        t.daemon = True
        t.start()

    def _run_pipeline(self, inputs: List[Path], out_dir: Path):
        self.run_btn.config(state="disabled")
        self.pb.config(maximum=len(inputs), value=0)

        ok = 0
        fail = 0
        last_err = ""

        ar_label = self.ar_var.get()
        keep_long = self.keep_long_var.get()
        up_choice = self.up_var.get()
        opt = self.opt_var.get()
        models = Path(self.models_entry.get().strip() or "models")
        mode = self.mode_var.get()

        for i, p in enumerate(inputs, start=1):
            success, msg = process_one(p, out_dir, ar_label, keep_long, up_choice, opt, models, mode)
            if success:
                ok += 1
            else:
                fail += 1
                last_err = msg

            # update progress and status
            self.pb.after(0, lambda v=i: self.pb.config(value=v))
            self.status.after(0, lambda i=i, p=str(p): self.status.config(text=f"{i}/{len(inputs)}: {p}"))

        def finish():
            self.run_btn.config(state="normal")
            txt = f"Готово. Успешно: {ok}, Ошибки: {fail}"
            if fail:
                txt += f" (last: {last_err})"
            messagebox.showinfo("Результат", txt)
            self.status.config(text=txt)
            # clear preview image reference
            self.preview_img = None
            self.preview_label.config(image="", text="No preview")

        self.after(0, finish)

    # ---------------- Preview logic ----------------
    def on_preview(self):
        """
        Preview the processed result for a single image:
        Priority:
         - if one file selected -> preview it
         - if in_entry contains a file path -> preview it
         - else -> show message
        """
        candidate = None
        if len(self.selected_files) == 1:
            candidate = Path(self.selected_files[0])
        else:
            txt = self.in_entry.get().strip()
            if txt and os.path.isfile(txt):
                candidate = Path(txt)

        if candidate is None:
            messagebox.showinfo("Preview", "Select a single file to preview (or choose a file in the input field).")
            return

        # build parameters
        ar_label = self.ar_var.get()
        keep_long = self.keep_long_var.get()
        up_choice = self.up_var.get()
        opt = self.opt_var.get()
        models = Path(self.models_entry.get().strip() or "models")
        mode = self.mode_var.get()

        # read and process (without writing)
        try:
            img = read_image(candidate)
            if img is None:
                messagebox.showerror("Error", "Cannot read selected image.")
                return
            h, w = img.shape[:2]
            tw, th = target_size(w, h, ar_label, keep_long)

            if mode == "Crop":
                cropped = center_crop_to_ratio(img, tw, th)
                processed = cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_LANCZOS4)
            else:
                processed = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LANCZOS4)

            ok, up, msg = upscale_if_needed(processed, models, up_choice)
            if not ok:
                messagebox.showerror("Preview error", msg)
                return

            # convert to PIL image for display, scale down to fit preview box
            rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            # fit into preview_label size (max 520x520)
            max_w, max_h = 520, 520
            pil.thumbnail((max_w, max_h), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil)
            self.preview_img = tk_img
            self.preview_label.config(image=tk_img, text="")
            # also show basic info in status
            self.status.config(text=f"Preview: {candidate.name} -> {tw}x{th}, mode={mode}, upscale={up_choice}")
        except Exception as e:
            messagebox.showerror("Preview failed", str(e))

    # ---------------- settings helpers ----------------
    def on_choose_files(self):
        files = filedialog.askopenfilenames(
            title="Выберите изображения",
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.webp;*.gif"), ("All", "*.*")]
        )
        if files:
            self.selected_files = list(files)
            self.in_entry.delete(0, tk.END)
            if len(files) == 1:
                self.in_entry.insert(0, files[0])
            else:
                self.in_entry.insert(0, f"[{len(files)} files selected]")

    # (the other handlers remain same; they were already defined earlier but overwritten above for preview - ensure they exist)
    def on_choose_input_dir(self):
        d = filedialog.askdirectory(title="Выберите папку")
        if d:
            self.selected_files = []
            self.in_entry.delete(0, tk.END)
            self.in_entry.insert(0, d)

    def on_choose_output_dir(self):
        d = filedialog.askdirectory(title="Папка сохранения")
        if d:
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, d)

    def on_choose_models_dir(self):
        d = filedialog.askdirectory(title="Папка моделей")
        if d:
            self.models_entry.delete(0, tk.END)
            self.models_entry.insert(0, d)


if __name__ == "__main__":
    App().mainloop()
