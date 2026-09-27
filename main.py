"""Real-time window capture + per-region card reading GUI, with online publishing/remote control.

Regions (ROIs) are named areas of the game window: one "board", one "hero" (my cards) and any
number of "player" regions (opponents). Each region is read on its own and reported under its
name. With no regions defined, the whole window is read and cards are split structurally.
"""
import base64
import ctypes
import json
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

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROI_FILE = os.path.join(APP_DIR, "rois.json")
DEBUG_IMAGE_PATH = os.path.join(APP_DIR, "debug_capture.png")
DEFAULT_ROI_SIZE = (300, 150)
PREVIEW_MAX_WIDTH = 720
PREVIEW_INTERVAL_MS = 3000
STABLE_SCANS = 2

KIND_NAMES = {"board": "보드", "hero": "내 카드", "player": "상대"}
KIND_COLORS = {"board": "#22e06b", "hero": "#3fa9ff", "player": "#ffb020"}

# Without this, Windows hands a scaled monitor's coordinates to tkinter, win32 and the
# screen grab in different units, so the boxes and the cropped regions disagree.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except (AttributeError, OSError):
    pass


def _load_rois():
    try:
        with open(ROI_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _roi_title(roi):
    return roi["name"] if roi["kind"] == "player" else KIND_NAMES[roi["kind"]]


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("카드 인식기")
        self.root.geometry("600x680")

        self.windows = []
        self.reader = CardReader()
        self.rois = _load_rois()
        self.overlays = {}
        self.overlays_visible = False

        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.running = False
        self.publisher = None

        self._build_ui()
        if self.reader.missing_suit_templates:
            self._log("문양 템플릿 없음: suit_templates/ 폴더에 "
                      + ", ".join(f"{n}.png" for n in self.reader.missing_suit_templates)
                      + " 를 넣고 다시 실행하세요. 그 문양은 인식되지 않습니다.")
        self._init_publisher()
        self.refresh_windows()
        self._refresh_roi_list()
        self._poll_log_queue()
        self.root.after(PREVIEW_INTERVAL_MS, self._live_preview_tick)

    def _init_publisher(self):
        if not Publisher.is_configured():
            self.publish_var.set(False)
            self._log("온라인 발행 키가 없어 로컬에서만 동작합니다.")
            return
        try:
            self.publisher = Publisher(
                on_error=lambda msg: self.log_queue.put(("log", msg)),
                on_command=lambda cmd: self.log_queue.put(("command", cmd)),
            )
            self._log("온라인 발행/원격 조작 준비 완료.")
        except Exception as exc:  # noqa: BLE001 — bad/missing key shouldn't stop local use
            self.publish_var.set(False)
            self._log(f"온라인 발행을 시작할 수 없습니다: {exc}")

    # ---------------------------------------------------------------- UI

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="감시할 창 선택").pack(anchor=tk.W)
        self.window_listbox = tk.Listbox(top, height=6, exportselection=False)
        self.window_listbox.pack(fill=tk.X, pady=4)
        self.window_listbox.bind("<<ListboxSelect>>", lambda _e: self._on_window_changed())
        ttk.Button(top, text="창 목록 새로고침", command=self.refresh_windows).pack(anchor=tk.W)

        roi_frame = ttk.LabelFrame(self.root, text="감시 영역 (ROI) — 없으면 창 전체를 읽습니다", padding=8)
        roi_frame.pack(fill=tk.X, padx=8, pady=4)
        self.roi_listbox = tk.Listbox(roi_frame, height=5, exportselection=False)
        self.roi_listbox.grid(row=0, column=0, columnspan=6, sticky="ew", pady=(0, 6))
        roi_frame.columnconfigure(0, weight=1)

        self.kind_var = tk.StringVar(value=KIND_NAMES["board"])
        ttk.Combobox(roi_frame, textvariable=self.kind_var, values=list(KIND_NAMES.values()),
                     state="readonly", width=8).grid(row=1, column=0, sticky="w")
        self.name_var = tk.StringVar(value="")
        ttk.Entry(roi_frame, textvariable=self.name_var, width=12).grid(row=1, column=1, padx=4)
        ttk.Label(roi_frame, text="(상대 이름)").grid(row=1, column=2, sticky="w")
        ttk.Button(roi_frame, text="영역 추가", command=self.add_roi_locally).grid(row=1, column=3, padx=4)
        ttk.Button(roi_frame, text="선택 삭제", command=self.delete_selected_roi).grid(row=1, column=4, padx=4)
        self.overlay_button = ttk.Button(roi_frame, text="박스 표시", command=self.toggle_overlays)
        self.overlay_button.grid(row=1, column=5, padx=4)

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
        ttk.Label(state_frame, text="현재:").pack(side=tk.LEFT)
        self.current_cards_var = tk.StringVar(value="(없음)")
        ttk.Label(state_frame, textvariable=self.current_cards_var, font=("Segoe UI", 11, "bold"),
                  wraplength=520).pack(side=tk.LEFT, padx=6)

        log_frame = ttk.Frame(self.root, padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)

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
                elif kind == "command":
                    self._handle_command(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    # ---------------------------------------------------------------- windows

    def refresh_windows(self):
        self.windows = [(h, t) for h, t in list_windows() if t != self.root.title()]
        self.window_listbox.delete(0, tk.END)
        for _, title in self.windows:
            self.window_listbox.insert(tk.END, title)
        if self.publisher:
            self.publisher.publish_windows(self.windows)
            self._publish_status()

    def _selected_hwnd(self):
        selection = self.window_listbox.curselection()
        if not selection:
            return None, None
        return self.windows[selection[0]]

    def _select_hwnd(self, hwnd):
        for i, (h, _) in enumerate(self.windows):
            if h == hwnd:
                self.window_listbox.selection_clear(0, tk.END)
                self.window_listbox.selection_set(i)
                self.window_listbox.see(i)
                self._on_window_changed()
                return
        self._log("선택한 창을 찾을 수 없습니다. 창 목록을 새로고침하세요.")

    def _on_window_changed(self):
        self._sync_overlays()
        self._publish_status()

    # ---------------------------------------------------------------- regions

    def _upsert_roi(self, roi):
        """Board and hero are single regions (fixed ids); players get their own ids."""
        kind = roi.get("kind") if roi.get("kind") in KIND_NAMES else "player"
        try:
            rect = {k: int(roi[k]) for k in ("x", "y", "w", "h")}
        except (KeyError, TypeError, ValueError):
            self._log("ROI 값이 올바르지 않습니다.")
            return
        if rect["w"] < 10 or rect["h"] < 10:
            self._log("ROI가 너무 작습니다.")
            return
        roi_id = kind if kind != "player" else (roi.get("id") or f"p{int(time.time() * 1000)}")
        name = (roi.get("name") or "").strip()[:20]
        if kind == "player" and not name:
            name = f"상대 {sum(r['kind'] == 'player' for r in self.rois) + 1}"
        new = {"id": roi_id, "kind": kind, "name": name, **rect}
        self.rois = [r for r in self.rois if r["id"] != roi_id] + [new]
        self._rois_changed()

    def _delete_roi(self, roi_id):
        self.rois = [r for r in self.rois if r["id"] != roi_id]
        overlay = self.overlays.pop(roi_id, None)
        if overlay:
            overlay.destroy()
        self._rois_changed()

    def _rois_changed(self):
        with open(ROI_FILE, "w", encoding="utf-8") as f:
            json.dump(self.rois, f, ensure_ascii=False, indent=1)
        self._refresh_roi_list()
        self._sync_overlays()
        self._publish_status()

    def _refresh_roi_list(self):
        order = {"board": 0, "hero": 1, "player": 2}
        self.rois.sort(key=lambda r: (order[r["kind"]], r["name"]))
        self.roi_listbox.delete(0, tk.END)
        for r in self.rois:
            self.roi_listbox.insert(tk.END, f"{_roi_title(r)}  ({r['x']},{r['y']}  {r['w']}×{r['h']})")

    def add_roi_locally(self):
        hwnd, _ = self._selected_hwnd()
        if hwnd is None:
            self._log("먼저 감시할 창을 목록에서 선택하세요.")
            return
        kind = next(k for k, v in KIND_NAMES.items() if v == self.kind_var.get())
        left, top, right, bottom = get_window_rect(hwnd)
        ox, oy = get_client_origin_screen(hwnd)
        w, h = DEFAULT_ROI_SIZE
        x = max(0, (right - left - w) // 2 + left - ox)
        y = max(0, (bottom - top - h) // 2 + top - oy)
        self._upsert_roi({"kind": kind, "name": self.name_var.get(), "x": x, "y": y, "w": w, "h": h})
        if not self.overlays_visible:
            self.toggle_overlays()
        self._log(f"'{KIND_NAMES[kind]}' 영역을 추가했습니다. 화면의 박스를 끌어서 위치를 맞추세요.")

    def delete_selected_roi(self):
        selection = self.roi_listbox.curselection()
        if selection:
            self._delete_roi(self.rois[selection[0]]["id"])

    def toggle_overlays(self):
        self.overlays_visible = not self.overlays_visible
        self.overlay_button.config(text="박스 숨기기" if self.overlays_visible else "박스 표시")
        self._sync_overlays()

    def _sync_overlays(self):
        """Keep one on-screen box per region, placed over the selected window."""
        hwnd, _ = self._selected_hwnd()
        if not self.overlays_visible or hwnd is None:
            for overlay in self.overlays.values():
                overlay.hide()
            return
        ox, oy = get_client_origin_screen(hwnd)
        for roi in self.rois:
            rect = (ox + roi["x"], oy + roi["y"], roi["w"], roi["h"])
            overlay = self.overlays.get(roi["id"])
            if overlay is None:
                overlay = RoiOverlay(self.root, rect, on_release=lambda *r, rid=roi["id"]: self._overlay_moved(rid, r))
                self.overlays[roi["id"]] = overlay
            overlay.set_rect(*rect)
            overlay.set_label(_roi_title(roi), KIND_COLORS[roi["kind"]])
            overlay.show()

    def _overlay_moved(self, roi_id, rect):
        hwnd, _ = self._selected_hwnd()
        roi = next((r for r in self.rois if r["id"] == roi_id), None)
        if hwnd is None or roi is None:
            return
        ox, oy = get_client_origin_screen(hwnd)
        x, y, w, h = rect
        self._upsert_roi({**roi, "x": x - ox, "y": y - oy, "w": w, "h": h})

    # ---------------------------------------------------------------- reading

    def _read_frame(self, frame):
        """{"board": [...], "hero": [...], "players": {name: [...]}} for one captured frame."""
        rois = list(self.rois)
        if not rois:
            table = self.reader.read_table(frame)
            return {"board": table["board"], "hero": table["hero"],
                    "players": {"기타": table["others"]} if table["others"] else {}}
        result = {"board": [], "hero": [], "players": {}}
        frame_h, frame_w = frame.shape[:2]
        for roi in rois:
            x0, y0 = max(0, roi["x"]), max(0, roi["y"])
            x1, y1 = min(frame_w, roi["x"] + roi["w"]), min(frame_h, roi["y"] + roi["h"])
            if x1 <= x0 or y1 <= y0:
                continue
            cards = [label for label, _, _ in self.reader.read(frame[y0:y1, x0:x1])]
            if roi["kind"] == "player":
                if cards:
                    result["players"][roi["name"]] = cards
            else:
                result[roi["kind"]] = cards
        return result

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
        self.running = True
        self._publish_status()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        scope = f"영역 {len(self.rois)}개" if self.rois else "창 전체"
        self._log(f"'{title}' 창 카드 인식을 시작합니다 ({scope}, 주기={interval_ms}ms).")

    def stop(self):
        self.stop_event.set()
        self.running = False
        self._publish_status()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self._log("감시를 정지했습니다.")

    def test_capture(self):
        """One-shot capture saved with every region boxed, and what each region reads."""
        hwnd, _ = self._selected_hwnd()
        if hwnd is None:
            self._log("먼저 감시할 창을 목록에서 선택하세요.")
            return
        frame = capture_window(hwnd)
        if frame is None:
            self._log("창을 캡처할 수 없습니다. 창이 최소화되어 있지 않은지 확인하세요.")
            return
        cv2.imwrite(DEBUG_IMAGE_PATH, self._annotate(frame))
        self._log(f"테스트 캡처 저장: {DEBUG_IMAGE_PATH} ({frame.shape[1]}x{frame.shape[0]})")
        self._log(_describe(self._read_frame(frame)))

    def _annotate(self, frame):
        out = frame.copy()
        for roi in self.rois:
            color = tuple(int(KIND_COLORS[roi["kind"]][i:i + 2], 16) for i in (5, 3, 1))  # hex → BGR
            cv2.rectangle(out, (roi["x"], roi["y"]), (roi["x"] + roi["w"], roi["y"] + roi["h"]), color, 2)
        return out

    def _capture_loop(self, hwnd, interval_ms):
        interval_sec = max(interval_ms, 50) / 1000.0
        last_error = None
        previous = None
        candidate, candidate_count = None, 0

        while not self.stop_event.is_set():
            frame = capture_window(hwnd)
            if frame is None:
                note = "창을 캡처할 수 없습니다. 창이 최소화되어 있지 않은지 확인하세요."
                if note != last_error:
                    self.log_queue.put(("log", note))
                    last_error = note
                time.sleep(interval_sec)
                continue
            last_error = None

            table = self._read_frame(frame)
            # Deal/flip animations show cards one at a time; only commit a state once it has
            # held for STABLE_SCANS consecutive scans so history doesn't fill with half-dealt boards.
            if table == candidate:
                candidate_count += 1
            else:
                candidate, candidate_count = table, 1
            if candidate_count >= STABLE_SCANS and table != previous:
                description = _describe(table)
                self.log_queue.put(("state", description))
                self.log_queue.put(("log", description))
                if self.publisher and self.publish_var.get():
                    self.publisher.publish(table, "영역" if self.rois else "창 전체")
                previous = table

            time.sleep(interval_sec)

    # ---------------------------------------------------------------- remote control

    def _handle_command(self, command):
        action = command.get("action")
        self._log(f"원격 명령: {action}")
        if action == "refresh_windows":
            self.refresh_windows()
        elif action == "select":
            self._select_hwnd(command.get("hwnd"))
        elif action == "preview":
            self._send_preview()
        elif action == "set_roi":
            self._upsert_roi(command.get("roi") or {})
            self._send_preview()
        elif action == "delete_roi":
            self._delete_roi(command.get("id"))
            self._send_preview()
        elif action == "start" and not self.running:
            self.start()
        elif action == "stop" and self.running:
            self.stop()
        self._publish_status()

    def _live_preview_tick(self):
        """While the owner has the page open, keep its preview fresh."""
        try:
            if self.publisher and self.publisher.viewer_active():
                self._send_preview(quiet=True)
        finally:
            self.root.after(PREVIEW_INTERVAL_MS, self._live_preview_tick)

    def _send_preview(self, quiet=False):
        hwnd, _ = self._selected_hwnd()
        if hwnd is None or not self.publisher:
            if not quiet:
                self._log("미리보기: 먼저 창을 선택하세요.")
            return
        frame = capture_window(hwnd)
        if frame is None:
            if not quiet:
                self._log("미리보기: 창을 캡처할 수 없습니다.")
            return
        height, width = frame.shape[:2]
        scale = min(1.0, PREVIEW_MAX_WIDTH / width)
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ok:
            self.publisher.publish_preview(base64.b64encode(jpeg.tobytes()).decode(), width, height)

    def _publish_status(self):
        if not self.publisher:
            return
        hwnd, title = self._selected_hwnd()
        self.publisher.publish_status({"running": self.running, "hwnd": hwnd, "title": title, "rois": self.rois})


def _describe(table):
    parts = []
    if table.get("board"):
        parts.append(f"보드: {' '.join(table['board'])}")
    if table.get("hero"):
        parts.append(f"내 카드: {' '.join(table['hero'])}")
    for name, cards in (table.get("players") or {}).items():
        parts.append(f"{name}: {' '.join(cards)}")
    return " | ".join(parts) if parts else "카드 없음"


def _another_instance_running():
    """Two copies would both answer the page's commands and overwrite each other's results,
    so hold a named Windows mutex for the lifetime of the process."""
    kernel32 = ctypes.windll.kernel32
    main._mutex = kernel32.CreateMutexW(None, False, "Local\\PokerCardReaderSingleInstance")
    return kernel32.GetLastError() == 183  # ERROR_ALREADY_EXISTS


def main():
    if _another_instance_running():
        ctypes.windll.user32.MessageBoxW(None, "카드 인식기가 이미 실행 중입니다. 기존 창을 사용하세요.", "카드 인식기", 0x40)
        return
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
