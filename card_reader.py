"""Reads face-up cards from a poker table image by their corner index (rank + suit).

Poker clients draw cards as white faces with the rank and suit stacked in the top-left
corner, and overlap hole cards so only that corner of the back card is visible. So:
  1. find white card-face regions,
  2. inside each, find rank glyphs along the top edge (one per card, even when overlapped)
     and the suit glyph directly beneath each,
  3. read the rank with OCR (works across clients' different fonts) and decide the suit
     by image matching against the four suit templates in suit_templates/.
Everything is measured relative to each glyph's own box, so window size doesn't matter.
"""
import os

import cv2
import numpy as np
from rapidocr import RapidOCR

SUIT_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "suit_templates")
SUIT_FILES = {"S": "spade", "H": "heart", "D": "diamond", "C": "club"}
SUIT_SIZE = (32, 32)
SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
RED_SUITS = {"H", "D"}
RANKS = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"}
MIN_OCR_SCORE = 0.5
MIN_SUIT_SCORE = 0.7
MIN_SUIT_MARGIN = 0.05


def _load_suit_templates():
    """Load suit_templates/{spade,heart,diamond,club}*.png — several per suit are allowed
    (e.g. club.png + club_prime.png) so one folder can cover different poker clients. Any
    size and padding works: each image is cropped to its symbol, like glyphs on screen."""
    templates, missing = [], []
    files = sorted(os.listdir(SUIT_TEMPLATE_DIR)) if os.path.isdir(SUIT_TEMPLATE_DIR) else []
    for code, name in SUIT_FILES.items():
        found = False
        for filename in files:
            stem, ext = os.path.splitext(filename.lower())
            if not stem.startswith(name) or ext not in (".png", ".jpg", ".jpeg", ".bmp"):
                continue
            img = _read_on_white(os.path.join(SUIT_TEMPLATE_DIR, filename))
            if img is None:
                continue
            # The symbol is the largest connected ink shape; stray marks (a bit of the rank
            # above it, the card border) in a loosely cropped template are ignored.
            n, _, stats, _ = cv2.connectedComponentsWithStats((img[:, :, 1] < 128).astype(np.uint8))
            if n < 2:
                continue
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            templates.append((code, _suit_vector(img, tuple(int(v) for v in stats[i, :4]))))
            found = True
        if not found:
            missing.append(name)
    return templates, missing


def _read_on_white(path):
    """Read an image as BGR, flattening any transparency onto white."""
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255
        return (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return img


def _normalize(vec):
    vec = vec - vec.mean()
    return vec / (np.linalg.norm(vec) + 1e-6)


def find_card_faces(img):
    """Bounding boxes of white card-face regions (overlapping cards merge into one)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 45)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_h = max(30, img.shape[0] * 0.05)
    faces = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # 0.35 rather than a stricter fill ratio: the active player's glowing white
        # name-plate outline merges into the hole-card face and lowers its fill.
        if h >= min_h and area > 0.35 * w * h and 0.4 < w / h < 2.2:
            faces.append((int(x), int(y), int(w), int(h)))
    return faces


def _ink_masks(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red = (hsv[:, :, 1] > 90) & ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 2] > 90)
    black = hsv[:, :, 2] < 110
    return red, black


def _corner_indices(img, face):
    """[(rank_bbox, suit_bbox, is_red)] in image coordinates for every card corner in a face."""
    x, y, w, h = face
    red, black = _ink_masks(img[y:y + h, x:x + w])
    found = []
    for is_red, mask in ((True, red), (False, black)):
        n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        comps = [tuple(int(v) for v in st[i]) for i in range(1, n) if st[i][4] > (h * h) * 0.0015]
        ranks = sorted(
            (c for c in comps if c[1] < h * 0.12 and h * 0.12 < c[3] < h * 0.35),
            key=lambda c: c[0],
        )
        merged = []  # "10" is two components side by side
        for c in ranks:
            if merged:
                px, py, pw, ph, pa = merged[-1]
                if c[0] - (px + pw) < h * 0.05 and abs(c[1] - py) < h * 0.05:
                    nx, ny = min(px, c[0]), min(py, c[1])
                    merged[-1] = (nx, ny, max(px + pw, c[0] + c[2]) - nx, max(py + ph, c[1] + c[3]) - ny, pa + c[4])
                    continue
            merged.append(c)
        for rx, ry, rw, rh, _ in merged:
            rcx = rx + rw / 2
            below = [
                c for c in comps
                if ry + rh * 0.8 < c[1] < ry + rh * 1.8 and abs(c[0] + c[2] / 2 - rcx) < rw and c[3] < rh * 1.2
            ]
            if below:
                s = min(below, key=lambda c: c[1])
                found.append(((x + rx, y + ry, rw, rh), (x + s[0], y + s[1], s[2], s[3]), is_red))
    return found


def _dark_ink(img, bbox):
    """Glyph pixels for red and black ink alike: both are dark in the green channel."""
    x, y, w, h = bbox
    return img[y:y + h, x:x + w, 1] < 128


def _ocr_input(img, bbox, height=64):
    """Upscale the grayscale glyph *before* thresholding: binarizing a ~12px glyph first
    destroys its shape, which made small Q read as A."""
    x, y, w, h = bbox
    margin = max(1, h // 6)
    gray = img[max(0, y - margin):y + h + margin, max(0, x - margin):x + w + margin, 1]
    gray = cv2.resize(gray, None, fx=height / h, fy=height / h, interpolation=cv2.INTER_CUBIC)
    _, glyph = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pad = height // 3
    glyph = cv2.copyMakeBorder(glyph, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(glyph, cv2.COLOR_GRAY2BGR)


def _normalize_rank(text):
    text = text.strip().upper().replace(" ", "")
    text = text.replace("O", "0").replace("I0", "10").replace("L0", "10").replace("1O", "10")
    return text if text in RANKS else None


def _is_red_ink(img, bbox):
    x, y, w, h = bbox
    patch = img[y:y + h, x:x + w].astype(int)
    ink = _dark_ink(img, bbox)
    if not ink.any():
        return False
    return float((patch[ink][:, 2] - patch[ink][:, 1]).mean()) > 60


def _suit_vector(img, bbox):
    mask = _dark_ink(img, bbox).astype(np.uint8) * 255
    return _normalize(cv2.resize(mask, SUIT_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32).ravel())


class CardReader:
    def __init__(self):
        self.suit_templates, self.missing_suit_templates = _load_suit_templates()
        self._ocr = RapidOCR()

    def read(self, img):
        """Return [(card_label, (x, y, w, h), score)] left-to-right, e.g. ("10♣", bbox, 0.93)."""
        return [card[:3] for card in _reading_order(self._read_with_faces(img))]

    def read_table(self, img):
        """Split visible cards into {"hero": [...], "board": [...], "others": [...]}.

        Positions shift between clients and layouts, so this uses structure: board cards
        are separate, equal-size faces in one row; hole cards overlap into a single face.
        The hero pair is the lowest overlapping pair — below the board when one is visible,
        since showdown pairs above it belong to opponents."""
        cards = self._read_with_faces(img)
        faces = {}
        for card in cards:
            faces.setdefault(card[3], []).append(card)

        # A face holding 2+ read cards is an overlapping pair even if it isn't wide: the
        # active player's glowing name plate can merge into it and make it taller.
        is_pair = {f: f[2] > f[3] or len(faces[f]) >= 2 for f in faces}
        board_faces = _largest_aligned_row([f for f in faces if not is_pair[f]])
        board_top = min((f[1] for f in board_faces), default=None)

        pairs = [f for f in faces if is_pair[f] and (board_top is None or f[1] > board_top)]
        hero_face = max(pairs, key=lambda f: f[1] + f[3], default=None)

        def labels(face_list):
            picked = [c for f in face_list for c in faces[f]]
            return [c[0] for c in _reading_order(picked)]

        others = [f for f in faces if f not in board_faces and f != hero_face]
        return {
            "hero": labels([hero_face] if hero_face else []),
            "board": labels(board_faces),
            "others": labels(others),
        }

    def _read_with_faces(self, img):
        if img is None or img.size == 0:
            return []
        cards = []
        for face in find_card_faces(img):
            for rank_box, suit_box, rank_is_red in _corner_indices(img, face):
                rank, rank_score = self._read_rank(img, rank_box)
                suit = self._read_suit(img, suit_box, rank_is_red)
                if rank and suit:
                    cards.append((f"{rank}{SUIT_SYMBOLS[suit]}", rank_box, rank_score, face))
        return cards

    def _read_rank(self, img, box):
        result = self._ocr(_ocr_input(img, box), use_det=False, use_cls=False, use_rec=True)
        if not result.txts or result.scores[0] < MIN_OCR_SCORE:
            return None, 0.0
        return _normalize_rank(result.txts[0]), float(result.scores[0])

    def _read_suit(self, img, box, rank_is_red):
        """Best-matching suit template, or None (skip the card) when the match is weak or the
        matched suit's colour disagrees with the rank's — e.g. a suit covered by a banner."""
        vec = _suit_vector(img, box)
        by_suit = {}
        for label, tmpl in self.suit_templates:
            by_suit[label] = max(by_suit.get(label, -1.0), float(vec @ tmpl))
        if not by_suit:
            return None
        ranked = sorted(by_suit.items(), key=lambda kv: -kv[1])
        best, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
        # Too close to call (spade vs club are similar blobs) → skip rather than guess.
        if best_score < MIN_SUIT_SCORE or best_score - runner_up < MIN_SUIT_MARGIN:
            return None
        if (best in RED_SUITS) != rank_is_red or _is_red_ink(img, box) != rank_is_red:
            return None
        return best


def _largest_aligned_row(faces):
    """Largest group (>= 2) of equal-height faces sharing a row — the community board."""
    best = []
    for anchor in faces:
        row = [f for f in faces
               if abs(f[1] - anchor[1]) < anchor[3] * 0.3 and abs(f[3] - anchor[3]) < anchor[3] * 0.15]
        if len(row) > len(best):
            best = row
    return best if len(best) >= 2 else []


def _reading_order(cards):
    """Top-to-bottom rows (cards whose rank glyphs overlap vertically), each left-to-right."""
    rows = []
    for card in sorted(cards, key=lambda c: c[1][1]):
        _, (_, y, _, h) = card[:2]
        if rows and y < rows[-1][0] + h * 0.5:
            rows[-1][1].append(card)
        else:
            rows.append([y, [card]])
    return [card for _, row in rows for card in sorted(row, key=lambda c: c[1][0])]
