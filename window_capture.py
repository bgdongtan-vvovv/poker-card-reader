"""EnumWindows-based window listing and PrintWindow-based frame capture."""
import ctypes

import numpy as np
import win32gui
import win32ui
from PIL import Image, ImageGrab

PW_CLIENTONLY = 1
PW_RENDERFULLCONTENT = 2


def list_windows():
    """Return a list of (hwnd, title) for visible top-level windows with a non-empty title."""
    windows = []

    def _enum_handler(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title.strip():
            return
        windows.append((hwnd, title))

    win32gui.EnumWindows(_enum_handler, None)
    return windows


def get_window_rect(hwnd):
    """Return (left, top, right, bottom) of the window in screen coordinates."""
    return win32gui.GetWindowRect(hwnd)


def get_client_origin_screen(hwnd):
    """Return the screen coordinates of the window's client-area origin (0, 0)."""
    return win32gui.ClientToScreen(hwnd, (0, 0))


def capture_window(hwnd):
    """Capture the client area of hwnd as a BGR numpy array, or None on failure.

    PrintWindow works even when the window is covered, but GPU-rendered game clients
    often come back all black through it — in that case fall back to grabbing the
    window's on-screen pixels (which requires the window to be visible)."""
    if not win32gui.IsWindow(hwnd):
        return None
    frame = _print_window(hwnd)
    if frame is None or frame.mean() < 3:
        frame = _grab_screen_region(hwnd)
    return frame


def _grab_screen_region(hwnd):
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    if right <= 0 or bottom <= 0:
        return None
    x, y = win32gui.ClientToScreen(hwnd, (0, 0))
    image = ImageGrab.grab(bbox=(x, y, x + right, y + bottom), all_screens=True)
    return np.array(image)[:, :, ::-1].copy()


def _print_window(hwnd):
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        return None

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()

    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(bitmap)

    try:
        result = ctypes.windll.user32.PrintWindow(
            hwnd, save_dc.GetSafeHdc(), PW_CLIENTONLY | PW_RENDERFULLCONTENT
        )
        if not result:
            return None

        bmp_info = bitmap.GetInfo()
        bmp_bits = bitmap.GetBitmapBits(True)
        image = Image.frombuffer(
            "RGB",
            (bmp_info["bmWidth"], bmp_info["bmHeight"]),
            bmp_bits, "raw", "BGRX", 0, 1,
        )
        return np.array(image)[:, :, ::-1].copy()  # RGB -> BGR for OpenCV
    finally:
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
