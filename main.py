"""Real-time window capture + ROI-limited card reading (corner rank/suit) GUI."""
import ctypes
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, ttk

import cv2

from card_reader import CardReader
from publisher import Publisher
from roi_overlay import RoiOverlay
from window_capture import (
    capture_window,
    get_client_origin_screen,
    get_window_rect,
    list_windows,
)

DEFAULT_ROI_SIZE = (300, 150)
DEBUG_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_capture.png")

# Without this, Windows hands a scaled monitor's coordinates to tkinter, win32 and the
# screen grab in different units, so the green box and the cropped region disagree.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except (AttributeError, OSError):
    pass


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("카드 인식기")
        self.root.geometry("560x600")

        self.windows = []
        self.reader = CardReader()
        self.roi_overlay = None
        self.roi_enabled = False

        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.publisher = None

        self._build_ui()
        self.refresh_windows()
        self._poll_log_queue()
        self._init_publisher()

    def _init_publisher(self):
        if not Publisher.is_configured():
            self.publish_var.set(False)
            self._log("온라인 발행 키가 없어 로컬에서만 동작합니다.")
            return
        try:
            self.publisher = Publisher(on_error=lambda msg: self.log_queue.put(("log", msg)))
            self._log("온라인 발행 준비 완료.")
        except Exception as exc:  # noqa: BLE001 — bad/missing key shouldn't stop local use
            self.publish_var.set(False)
            self._log(f"온라인 발행을 시작할 수 없습니다: {exc}")

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.BOTH, expand=False)

        ttk.Label(top, text="감시할 창 선택").pack(anchor=tk.W)
        self.window_listbox = tk.Listbox(top, height=8)
        self.window_listbox.pack(fill=tk.X, pady=4)

        ttk.Button(top, text="창 목록 새로고침", command=self.refresh_windows).pack(anchor=tk.W)

        roi_frame = ttk.LabelFrame(self.root, text="ROI (감시 영역)", padding=8)
        roi_frame.pack(fill=tk.X, padx=8, pady=4)

        self.roi_toggle_button = ttk.Button(roi_frame, text="ROI 표시", command=self.toggle_roi)
        self.roi_toggle_button.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 6))

        self.roi_x_var = tk.StringVar(value="0")
        self.roi_y_var = tk.StringVar(value="0")
        self.roi_w_var = tk.StringVar(value=str(DEFAULT_ROI_SIZE[0]))
        self.roi_h_var = tk.StringVar(value=str(DEFAULT_ROI_SIZE[1]))

        for i, (label, var) in enumerate(
            [("X", self.roi_x_var), ("Y", self.roi_y_var), ("W", self.roi_w_var), ("H", self.roi_h_var)]
        ):
            ttk.Label(roi_frame, text=label).grid(row=1, column=i * 2, sticky=tk.W)
            ttk.Entry(roi_frame, textvariable=var, width=6).grid(row=1, column=i * 2 + 1, padx=(2, 8))

        ttk.Button(roi_frame, text="적용", command=self.apply_roi_fields).grid(row=1, column=8, padx=(4, 0))

        options = ttk.Frame(self.root, padding=8)
        options.pack(fill=tk.X)
        ttk.Label(options, text="스캔 주기(ms)").grid(row=0, column=0, sticky=tk.W)
        self.interval_var = tk.StringVar(value="300")
        ttk.Entry(options, textvariable=self.interval_var, width=8).grid(row=0, column=1, padx=4)

        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill=tk.X)
        self.start_button = ttk.Button(controls, text="시작", command=self.start)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(controls, text="정지", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="테스트 캡처", command=self.test_capture).pack(side=tk.LEFT, padx=4)
        self.publish_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="온라인 발행", variable=self.publish_var).pack(side=tk.LEFT, padx=8)

        state_frame = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        state_frame.pack(fill=tk.X)
        ttk.Label(state_frame, text="현재 카드:").pack(side=tk.LEFT)
        self.current_cards_var = tk.StringVar(value="(없음)")
        ttk.Label(state_frame, textvariable=self.current_cards_var, font=("Segoe UI", 12, "bold")).pack(
            side=tk.LEFT, padx=6
        )

        log_frame = ttk.Frame(self.root, padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def refresh_windows(self):
        self.windows = list_windows()
        self.window_listbox.delete(0, tk.END)
        for _, title in self.windows:
            self.window_listbox.insert(tk.END, title)

    def _selected_hwnd(self):
        selection = self.window_listbox.curselection()
        if not selection:
            return None, None
        return self.windows[selection[0]]

    def toggle_roi(self):
        if self.roi_enabled:
            self.roi_enabled = False
            if self.roi_overlay:
                self.roi_overlay.hide()
            self.roi_toggle_button.config(text="ROI 표시")
            return

        hwnd, _ = self._selected_hwnd()
        if hwnd is None:
            self._log("먼저 감시할 창을 목록에서 선택하세요.")
            return

        if self.roi_overlay is None:
            left, top, right, bottom = get_window_rect(hwnd)
            win_w, win_h = right - left, bottom - top
            w, h = DEFAULT_ROI_SIZE
            x = left + max(0, (win_w - w) // 2)
            y = top + max(0, (win_h - h) // 2)
            self.roi_overlay = RoiOverlay(self.root, (x, y, w, h), on_change=self._set_roi_fields)
            self._set_roi_fields(x, y, w, h)

        self.roi_enabled = True
        self.roi_overlay.show()
        self.roi_toggle_button.config(text="ROI 숨기기")

    def _set_roi_fields(self, x, y, w, h):
        self.roi_x_var.set(str(int(x)))
        self.roi_y_var.set(str(int(y)))
        self.roi_w_var.set(str(int(w)))
        self.roi_h_var.set(str(int(h)))

    def apply_roi_fields(self):
        if self.roi_overlay is None:
            self._log("먼저 'ROI 표시'로 박스를 띄우세요.")
            return
        try:
            x, y, w, h = (int(v.get()) for v in (self.roi_x_var, self.roi_y_var, self.roi_w_var, self.roi_h_var))
        except ValueError:
            self._log("ROI 값이 올바르지 않습니다.")
            return
        self.roi_overlay.set_rect(x, y, w, h)

    def _log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _poll_log_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "state":
                    self.current_cards_var.set(payload if payload else "(없음)")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _grab_target(self, hwnd):
        """Capture the window and crop it to the ROI if one is active. Returns (image, note) or (None, reason)."""
        frame = capture_window(hwnd)
        if frame is None:
            return None, "창을 캡처할 수 없습니다. 창이 최소화되어 있지 않은지 확인하세요."
        if not (self.roi_enabled and self.roi_overlay is not None):
            return frame, "창 전체"

        rx, ry, rw, rh = self.roi_overlay.get_rect()
        ox, oy = get_client_origin_screen(hwnd)
        cx, cy = rx - ox, ry - oy
        frame_h, frame_w = frame.shape[:2]
        x0, y0 = max(0, cx), max(0, cy)
        x1, y1 = min(frame_w, cx + rw), min(frame_h, cy + rh)
        if x1 <= x0 or y1 <= y0:
            return None, "ROI가 창 영역을 벗어나 있습니다."
        return frame[y0:y1, x0:x1], "ROI"

    def start(self):
        hwnd, title = self._selected_hwnd()
        if hwnd is None:
            self._log("먼저 감시할 창을 목록에서 선택하세요.")
            return
        try:
            interval_ms = int(self.interval_var.get())
        except ValueError:
            self._log("스캔 주기 값이 올바르지 않습니다.")
            return

        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._capture_loop, args=(hwnd, interval_ms), daemon=True)
        self.worker_thread.start()

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        roi_note = "ROI 적용" if self.roi_enabled else "창 전체"
        self._log(f"'{title}' 창 카드 인식을 시작합니다 ({roi_note}, 주기={interval_ms}ms).")

    def stop(self):
        self.stop_event.set()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self._log("감시를 정지했습니다.")

    def test_capture(self):
        """One-shot capture that saves what the reader sees, with recognized cards boxed."""
        hwnd, _ = self._selected_hwnd()
        if hwnd is None:
            self._log("먼저 감시할 창을 목록에서 선택하세요.")
            return
        target, note = self._grab_target(hwnd)
        if target is None:
            self._log(note)
            return

        cards = self.reader.read(target)
        annotated = target.copy()
        for label, (x, y, w, h), _ in cards:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(DEBUG_IMAGE_PATH, annotated)

        self._log(f"테스트 캡처 저장: {DEBUG_IMAGE_PATH} ({note}, {target.shape[1]}x{target.shape[0]})")
        if cards:
            self._log("인식: " + ", ".join(f"{label}({score:.2f})" for label, _, score in cards))
            self._log(_describe(self.reader.read_table(target)))
        else:
            self._log("인식된 카드 없음 — 저장된 이미지에 카드가 보이는지 확인하세요.")

    def _capture_loop(self, hwnd, interval_ms):
        interval_sec = max(interval_ms, 50) / 1000.0
        last_error = None
        previous = None

        while not self.stop_event.is_set():
            target, note = self._grab_target(hwnd)
            if target is None:
                if note != last_error:
                    self.log_queue.put(("log", note))
                    last_error = note
                time.sleep(interval_sec)
                continue
            last_error = None

            table = self.reader.read_table(target)
            if table != previous:
                description = _describe(table)
                self.log_queue.put(("state", description))
                self.log_queue.put(("log", description))
                if self.publisher and self.publish_var.get():
                    self.publisher.publish(table, note)
                previous = table

            time.sleep(interval_sec)


def _describe(table):
    parts = [f"{name}: {' '.join(table[key])}" for key, name in
             (("hero", "내 카드"), ("board", "보드"), ("others", "기타")) if table[key]]
    return " | ".join(parts) if parts else "카드 없음"


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
