import math
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Return an uint8 image in [0,255], accepting float [0,1] or [0,255]."""
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def _to_hsv(img_u8: np.ndarray, input_space: str) -> np.ndarray:
    if input_space.upper() == "RGB":
        return cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    if input_space.upper() == "BGR":
        return cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)
    raise ValueError('input_space must be "RGB" or "BGR".')


def find_blue_square_hsv(
    img: np.ndarray,
    input_space: str = "RGB",
    h: Tuple[int, int] = (100, 130),
    s_min: int = 20,
    v_min: int = 20,
    min_area: int = 900,
    max_area: Optional[int] = 15000,
    square_tol: float = 1.50,
    extent_min: float = 0.35,
    morph_ksize: int = 5,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """
    Detect a blue-ish square in the image.

    Returns:
        mask (uint8): binary mask.
        info (dict|None): best candidate's geometry, or None if not found.
    """
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must be an HxWx3 image array.")

    img_u8 = _ensure_uint8(img)
    hsv = _to_hsv(img_u8, input_space)

    lower = np.array([h[0], s_min, v_min], dtype=np.uint8)
    upper = np.array([h[1], 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=5)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask, None

    best: Optional[Dict[str, Any]] = None
    best_score = -1.0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        rect = cv2.minAreaRect(c)
        (cxr, cyr), (wr, hr), angle = rect
        if wr < 1e-6 or hr < 1e-6:
            continue

        ar = max(wr, hr) / (min(wr, hr) + 1e-9)
        if abs(ar - 1.0) > square_tol:
            continue

        extent = area / (wr * hr + 1e-9)
        if extent < extent_min:
            continue

        x, y, w, h2 = cv2.boundingRect(c)
        M = cv2.moments(c)
        cx = M["m10"] / (M["m00"] + 1e-9)  # float
        cy = M["m01"] / (M["m00"] + 1e-9)

        score = ((math.log(area + 1)) ** 0.2) * ((1.0 / (abs(ar - 1.0) + 1e-3)) ** 8) * (extent ** 2)
        if score > best_score:
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            best = {
                "area": float(area),
                "center": (cx, cy),
                "bbox_axis": (int(x), int(y), int(w), int(h2)),
                "rect_rot": ((float(cxr), float(cyr)), (float(wr), float(hr)), float(angle)),
                "box_points": box,
                "contour": c,
                "ar": float(ar),
                "extent": float(extent),
            }
            best_score = score

    return mask, best


def find_red_square_hsv(
    img: np.ndarray,
    input_space: str = "RGB",
    h1: Tuple[int, int] = (0, 10),
    h2: Tuple[int, int] = (170, 179),
    s_min: int = 20,
    v_min: int = 20,
    min_area: int = 900,
    max_area: Optional[int] = 15000,
    square_tol: float = 1.50,
    extent_min: float = 0.35,
    morph_ksize: int = 5,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """
    Detect a red-ish square in the image (two hue ranges to wrap around 0).
    """
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must be an HxWx3 image array.")

    img_u8 = _ensure_uint8(img)
    hsv = _to_hsv(img_u8, input_space)

    lower1 = np.array([h1[0], s_min, v_min], dtype=np.uint8)
    upper1 = np.array([h1[1], 255, 255], dtype=np.uint8)
    lower2 = np.array([h2[0], s_min, v_min], dtype=np.uint8)
    upper2 = np.array([h2[1], 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=5)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask, None

    best: Optional[Dict[str, Any]] = None
    best_score = -1.0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
        (cxr, cyr), (wr, hr), angle = rect
        if wr < 1e-6 or hr < 1e-6:
            continue

        ar = max(wr, hr) / (min(wr, hr) + 1e-9)
        if abs(ar - 1.0) > square_tol:
            continue

        extent = area / (wr * hr + 1e-9)
        if extent < extent_min:
            continue

        x, y, w, h = cv2.boundingRect(c)

        M = cv2.moments(c)
        cx = int(M["m10"] / (M["m00"] + 1e-9))
        cy = int(M["m01"] / (M["m00"] + 1e-9))

        score = ((math.log(area + 1)) ** 0.2) * ((1.0 / (abs(ar - 1.0) + 1e-3)) ** 8) * (extent ** 2)
        if score > best_score:
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            best = {
                "area": float(area),
                "center": (cx, cy),
                "bbox_axis": (int(x), int(y), int(w), int(h)),
                "rect_rot": ((float(cxr), float(cyr)), (float(wr), float(hr)), float(angle)),
                "box_points": box,
                "contour": c,
                "ar": float(ar),
                "extent": float(extent),
            }
            best_score = score

    return mask, best


color_thresholds = {
    "red": ((150, 255), (0, 170), (0, 110)),   # R, G, B
    "blue": ((0, 120), (0, 120), (150, 255)),  # R, G, B
}


def normal_search(current_image_rgb: np.ndarray, color: str, min_area: int = 80):
    """Fast RGB threshold search (no HSV / no Retinex)."""
    R, G, B = color_thresholds[color]
    lower = np.array([R[0], G[0], B[0]])
    upper = np.array([R[1], G[1], B[1]])
    mask = cv2.inRange(current_image_rgb, lower, upper)

    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask, None

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return mask, None

    x, y, w, h = cv2.boundingRect(c)
    M = cv2.moments(c)
    cx = int(M["m10"] / (M["m00"] + 1e-9))
    cy = int(M["m01"] / (M["m00"] + 1e-9))

    return mask, {"area": area, "bbox": (x, y, w, h), "center": (cx, cy), "contour": c}


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detect.py input.jpg [output.jpg]")
        sys.exit(0)

    filename = sys.argv[1]
    base_in = "R"
    base_out = "detect"
    os.makedirs(base_out, exist_ok=True)

    in_path = os.path.join(base_in, filename)
    out_path = os.path.join(base_out, filename)

    img = np.asarray(Image.open(in_path).convert("RGB"), dtype=np.uint8)

    t_total_start = time.perf_counter()
    t_det_start = time.perf_counter()
    mask, info = find_blue_square_hsv(img=img, input_space="RGB")
    t_det_end = time.perf_counter()
    t_total_end = time.perf_counter()

    print(f"[Timing] detect: {(t_det_end - t_det_start) * 1000:.2f} ms")
    print(f"[Timing] total: {(t_total_end - t_total_start) * 1000:.2f} ms")
    print(info)

    Image.fromarray(mask).save(out_path)

    if info is not None:
        vis = img.copy()
        x, y, w, h = info["bbox_axis"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(base_out, "vis_" + filename), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
