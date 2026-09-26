"""Reads face-up cards from a poker table image by their corner index (rank + suit).

The client draws cards as white faces with the rank and suit stacked in the top-left
corner, and overlaps hole cards so only that corner of the back card is visible. So:
  1. find white card-face regions,
  2. inside each, find rank glyphs along the top edge (one per card, even when overlapped)
     and the suit glyph directly beneath each,
  3. normalize each glyph to a fixed size and classify it against real glyphs cut from
     actual client screenshots (glyphs/rank_*.png, glyphs/suit_*.png).
Normalizing to the glyph's own bounding box makes this independent of window size.
"""
import glob
import os

import cv2
import numpy as np

GLYPH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glyphs")
RANK_SIZE = (32, 40)
SUIT_SIZE = (32, 32)
SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}

# Enclosed holes in each rank glyph of the client font (verified on every sample).
# Pixel correlation alone nearly ties 8 vs 9 (their shapes differ by one small gap),
# but the hole count separates them cleanly.
RANK_HOLES = {"8": 2, "4": 1, "6": 1, "9": 1, "A": 1, "Q": 1, "10": 1,
              "2": 0, "3": 0, "5": 0, "7": 0, "J": 0, "K": 0}


def _load_templates(prefix):
    templates = []
    for path in glob.glob(os.path.join(GLYPH_DIR, f"{prefix}_*.png")):
        label = os.path.basename(path).split("_")[1]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates.append((label, _normalize(img.astype(np.float32).ravel())))
    return templates


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
            faces.append((x, y, w, h))
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
        comps = [tuple(st[i]) for i in range(1, n) if st[i][4] > (h * h) * 0.0015]
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


def _glyph_vector(img, bbox, is_red, size):
    x, y, w, h = bbox
    red, black = _ink_masks(img[y:y + h, x:x + w])
    mask = (red if is_red else black).astype(np.uint8) * 255
    return _normalize(cv2.resize(mask, size, interpolation=cv2.INTER_AREA).astype(np.float32).ravel())


def _count_holes(img, bbox):
    x, y, w, h = bbox
    ink = np.pad((img[y:y + h, x:x + w, 1] < 128).astype(np.uint8), 1)
    n, _ = cv2.connectedComponents(1 - ink, connectivity=4)
    return n - 2  # drop the ink label and the outer background region


def _classify(vec, templates, allowed=None):
    best_label, best_score = None, -1.0
    for label, tmpl in templates:
        if allowed is not None and label not in allowed:
            continue
        score = float(vec @ tmpl)
        if score > best_score:
            best_label, best_score = label, score
    return best_label, best_score


class CardReader:
    # On held-out real screenshots, correct suits scored >= 0.90 while a suit hidden by
    # the "WIN" banner scored 0.79 — so a strict suit bar rejects occluded cards instead
    # of misreading them.
    def __init__(self, min_rank_score=0.6, min_suit_score=0.85):
        self.min_rank_score = min_rank_score
        self.min_suit_score = min_suit_score
        self.rank_templates = _load_templates("rank")
        self.suit_templates = _load_templates("suit")

    def read(self, img):
        """Return [(card_label, (x, y, w, h), score)] left-to-right, e.g. ("10♣", bbox, 0.93)."""
        return [card[:3] for card in _reading_order(self._read_with_faces(img))]

    def read_table(self, img):
        """Split visible cards into {"hero": [...], "board": [...], "others": [...]}.

        Positions shift between desktop/mobile/scrolled layouts, so this uses structure:
        board cards are separate, equal-size faces in one row; hole cards overlap into a
        single wide face. The hero pair is the lowest overlapping pair — below the board
        when one is visible, since showdown pairs above it belong to opponents."""
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
        if img is None or img.size == 0 or not self.rank_templates:
            return []
        cards = []
        for face in find_card_faces(img):
            face = tuple(int(v) for v in face)
            for rank_box, suit_box, is_red in _corner_indices(img, face):
                holes = _count_holes(img, rank_box)
                allowed = {r for r, n in RANK_HOLES.items() if n == holes} or None
                rank, rank_score = _classify(
                    _glyph_vector(img, rank_box, is_red, RANK_SIZE), self.rank_templates, allowed
                )
                suit, suit_score = _classify(_glyph_vector(img, suit_box, is_red, SUIT_SIZE), self.suit_templates)
                if rank_score >= self.min_rank_score and suit_score >= self.min_suit_score:
                    label = f"{rank}{SUIT_SYMBOLS[suit]}"
                    cards.append((label, rank_box, min(rank_score, suit_score), face))
        return cards


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
