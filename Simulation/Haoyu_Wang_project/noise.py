from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import os

ROI = Tuple[int, int, int, int]  # (x0, y0, w, h)

def _rgb_to_luma_y(img_rgb01: np.ndarray) -> np.ndarray:
    if img_rgb01.ndim != 3 or img_rgb01.shape[2] != 3:
        raise ValueError("img_rgb01 must be HxWx3 RGB.")
    img = img_rgb01.astype(np.float32, copy=False)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b  # Rec.601
    return y.astype(np.float32, copy=False)

def _apply_roi_and_mask_2d(y: np.ndarray,
                           roi: Optional[ROI] = None,
                           mask: Optional[np.ndarray] = None) -> np.ndarray:
    if y.ndim != 2:
        raise ValueError("y must be HxW.")
    H, W = y.shape
    if roi is not None:
        x0, y0, w, h = roi
        x1, y1 = x0 + w, y0 + h
        if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
            raise ValueError("roi out of bounds.")
        y = y[y0:y1, x0:x1]
        if mask is not None:
            mask = mask[y0:y1, x0:x1]
    if mask is not None:
        m = mask.astype(bool)
        vals = y[m]
    else:
        vals = y.reshape(-1)
    if vals.size == 0:
        raise ValueError("ROI/mask selects zero pixels.")
    return vals

def noise_sigma_mad_single(img_rgb01: np.ndarray,
                           roi: Optional[ROI] = None,
                           mask: Optional[np.ndarray] = None,
                           blur_ksize: int = 5,
                           blur_sigma: float = 1.0) -> float:
    """
    Estimate noise level sigma from a single image using MAD on high-pass residual.
    Assumes (roughly) additive noise; ROI should be flat/textureless for best accuracy.

    Returns:
      sigma in the same scale as input (since input is [0,1], sigma is also in [0,1] units).
    """
    y = _rgb_to_luma_y(img_rgb01)

    # Low-pass (structure) + residual (high-pass)
    # ksize must be odd
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    y_lp = cv2.GaussianBlur(y, (blur_ksize, blur_ksize), blur_sigma)
    r = y - y_lp  # residual

    vals = _apply_roi_and_mask_2d(r, roi=roi, mask=mask)

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad  # MAD -> sigma for Gaussian noise
    return sigma

def noise_hf_laplacian_single(img_rgb01: np.ndarray,
                              roi: Optional[ROI] = None,
                              mask: Optional[np.ndarray] = None,
                              ksize: int = 3) -> float:
    """
    High-frequency energy proxy: mean(abs(Laplacian(Y))) over ROI/mask.
    Larger means more high-frequency content (noise/texture/sharpening artifacts).
    """
    y = _rgb_to_luma_y(img_rgb01)
    lap = cv2.Laplacian(y, ddepth=cv2.CV_32F, ksize=ksize)
    a = np.abs(lap)
    vals = _apply_roi_and_mask_2d(a, roi=roi, mask=mask)
    return float(np.mean(vals))

def noise_sigma_rms_residual_single(img_rgb01: np.ndarray, roi: ROI, blur_ksize: int = 5, blur_sigma: float = 1.0) -> float:
    y = _rgb_to_luma_y(img_rgb01)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    y_lp = cv2.GaussianBlur(y, (blur_ksize, blur_ksize), blur_sigma)
    r = y - y_lp

    x0, y0, w, h = roi
    rr = r[y0:y0+h, x0:x0+w].reshape(-1)
    return float(np.sqrt(np.mean(rr * rr)))

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))  
    name1 = "frames" 
    print(name1)
    folder = os.path.join(here, name1)              

    print(f"{'Image Name':<20} | {'noise_sigma_mad_single':<20}| {'hf':<20}| {'noise_sigma_rms_residual_single':<20}")
    print("-" * 45)

    for name in os.listdir(folder):
        
        full = os.path.join(folder, name)

        img_raw = Image.open(full).convert("RGB")

        img = np.asarray(img_raw).astype(np.float32) / 255.0  
        
        sigma = noise_sigma_mad_single(img, roi=(50, 300, 200, 120))
        hf = noise_hf_laplacian_single(img, roi=(50, 300, 200, 120))
        rms = noise_sigma_rms_residual_single(img, roi=(50, 300, 200, 120))
        print(f"{name:<20} : {sigma:.6f} : {hf:.6f}: {rms:.6f}")

        