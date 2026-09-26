"""A draggable, resizable green-bordered overlay window that marks a screen region (ROI).
Everything inside the border stays click-through and visible (Windows -transparentcolor),
so it can float over the target window without blocking it."""
import tkinter as tk

TRANSPARENT_COLOR = "magenta"
BORDER_COLOR = "#00ff00"
BORDER_WIDTH = 3
HANDLE_SIZE = 14
MIN_SIZE = 40


class RoiOverlay:
    def __init__(self, root, rect, on_change=None):
        """rect: (x, y, w, h) in screen coordinates. on_change: callback(x, y, w, h)."""
        self.on_change = on_change
        self.x, self.y, self.w, self.h = rect

        self.top = tk.Toplevel(root)
        self.top.overrideredirect(True)
        self.top.attributes("-topmost", True)
        self.top.attributes("-transparentcolor", TRANSPARENT_COLOR)
        self.top.configure(bg=TRANSPARENT_COLOR)

        self.canvas = tk.Canvas(self.top, bg=TRANSPARENT_COLOR, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._drag_mode = None  # "move" or "resize"
        self._drag_start = (0, 0)
        self._drag_origin_rect = (0, 0, 0, 0)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        self._apply_geometry()
        self._redraw()
        self.top.withdraw()

    def show(self):
        self._apply_geometry()
        self._redraw()
        self.top.deiconify()

    def hide(self):
        self.top.withdraw()

    def destroy(self):
        self.top.destroy()

    def get_rect(self):
        return (self.x, self.y, self.w, self.h)

    def set_rect(self, x, y, w, h):
        self.x, self.y = int(x), int(y)
        self.w, self.h = max(MIN_SIZE, int(w)), max(MIN_SIZE, int(h))
        self._apply_geometry()
        self._redraw()

    def _apply_geometry(self):
        self.top.geometry(f"{self.w}x{self.h}+{self.x}+{self.y}")

    def _redraw(self):
        self.canvas.delete("all")
        half = BORDER_WIDTH / 2
        self.canvas.create_rectangle(
            half, half, self.w - half, self.h - half,
            outline=BORDER_COLOR, width=BORDER_WIDTH,
        )
        self.canvas.create_rectangle(
            self.w - HANDLE_SIZE, self.h - HANDLE_SIZE, self.w, self.h,
            fill=BORDER_COLOR, outline=BORDER_COLOR, tags="handle",
        )

    def _on_press(self, event):
        if self.w - event.x <= HANDLE_SIZE and self.h - event.y <= HANDLE_SIZE:
            self._drag_mode = "resize"
        else:
            self._drag_mode = "move"
        self._drag_start = (event.x_root, event.y_root)
        self._drag_origin_rect = (self.x, self.y, self.w, self.h)

    def _on_drag(self, event):
        if self._drag_mode is None:
            return
        dx = event.x_root - self._drag_start[0]
        dy = event.y_root - self._drag_start[1]
        ox, oy, ow, oh = self._drag_origin_rect

        if self._drag_mode == "move":
            self.x, self.y = ox + dx, oy + dy
            self._apply_geometry()
        else:  # resize
            self.w = max(MIN_SIZE, ow + dx)
            self.h = max(MIN_SIZE, oh + dy)
            self._apply_geometry()
            self._redraw()

        if self.on_change:
            self.on_change(self.x, self.y, self.w, self.h)

    def _on_release(self, _event):
        self._drag_mode = None
