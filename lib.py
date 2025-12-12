import numpy as np
from PIL import Image
from typing import Tuple
import time 
from scipy.fft import rfft2, irfft2
import os
import sys

def grad_forward(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward difference gradient with Neumann boundary (zero-gradient at border).
    u: (H, W)
    return: (gx, gy) same shape
    """
    gx = np.zeros_like(u)
    gy = np.zeros_like(u)

    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gx[:, -1] = 0.0

    gy[:-1, :] = u[1:, :] - u[:-1, :]
    gy[-1, :] = 0.0

    return gx, gy

# ----------------------
# shrinkage / soft-threshold
# ----------------------
def shrink(x: np.ndarray, lam: float) -> np.ndarray:
    """
    shrink(x, lam) = x/|x| * max(|x| - lam, 0)
    """
    mag = np.abs(x)
    # avoid division by zero
    scale = np.maximum(mag - lam, 0.0) / (mag + np.finfo(x.dtype).eps)
    return x * scale

def psf2otf(psf, outSize):
    """
    Convert a spatial-domain convolution kernel (PSF) into its frequency-domain
    representation (OTF), with automatic circular shifting.

    Args:
        psf: The point spread function (e.g., a 3×3 Laplacian kernel).
        outSize: Output size (H, W), typically the size of the target image.

    Returns:
        otf: A complex-valued array of shape outSize representing the OTF.
    """
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    
    padSize = outSize - psfSize
    
    psf_padded = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), 'constant')
    
    # 2. 循环移位 (Circular Shift)
    # 我们要把 PSF 的中心移动到 (0,0)
    # 假设 PSF 中心在 psfSize / 2
    # 我们需要向左、向上滚动这些距离
    
    # 计算需要滚动的距离 (负数表示向左/上滚)
    shift = -(psfSize // 2)
    
    # 使用 np.roll 进行循环移位
    psf_shifted = np.roll(psf_padded, shift=shift, axis=(0, 1))
    
    # 3. 计算 FFT
    otf = rfft2(psf_shifted,workers=-1)
    
    return otf


def decompose_single_channel(
    S: np.ndarray,
    c1: float = 0.02,
    c2: float = 5.0,
    lam: float = 10.0,
    eps_1: float = 6e-3,
    eps_2: float = 1e-3,
    max_outer_iter: int = 50,
    min_iter = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a single image channel into reflectance and illumination
    using the weighted variational Retinex model with ADMM optimization.

    Args:
        S: Input image channel in [0, 1], shape (H, W).
        c1: Regularization weight for the reflectance TV term.
        c2: Regularization weight for the illumination smoothness term.
        lam: ADMM penalty parameter λ.
        eps_1: Relative stopping tolerance for the reflectance update.
        eps_2: Relative stopping tolerance for the illumination update.
        max_outer_iter: Maximum number of ADMM outer iterations.
        min_iter: Minimum number of iterations before early stopping
            is allowed.

    Returns:
        R: Estimated reflectance component in [0, 1], shape (H, W).
        L: Estimated illumination component in [0, 1], shape (H, W).
    """

    eps = 1e-6
    s = np.log(np.clip(S, eps, 1.0))  # log-domain avoid log 0

    # r^0 = 0 (R^0 = 1), l^0 = s (L^0 = S)
    r = np.zeros_like(s)
    l = s.copy()

    bh = np.zeros_like(s)
    bv = np.zeros_like(s)

    r_prev = r.copy()
    l_prev = l.copy()
    bh_prev = bh.copy()
    bv_prev = bv.copy()

    R_prev = np.exp(r_prev)
    L_prev = np.exp(l_prev)

    kernel_h = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]) 
    kernel_v = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
    H, W = s.shape
    F_Dh = psf2otf(kernel_h, (H, W))  
    F_Dv = psf2otf(kernel_v, (H, W))
    Laplacian_response = np.abs(F_Dh) ** 2 + np.abs(F_Dv) ** 2

    iter_count = 0
    for it in range(max_outer_iter):
        iter_count +=1
        # ---------- P1: update d (soft-threshold) ----------
        r_h_prev, r_v_prev = grad_forward(r_prev)
        # element-wise weighted gradient: R^{k-1} · ∇r^{k-1}

        dh_cur = shrink(R_prev * r_h_prev + bh_prev, 1.0 / (2.0 * lam))
        dv_cur = shrink(R_prev * r_v_prev + bv_prev, 1.0 / (2.0 * lam))

        # ---------- P2: update r via FFT ----------

        R_scaler_sq = np.mean(R_prev) #incorrect in math but adapted for engineering
        s_hat = s - l_prev
        coeff = c1 * lam * R_scaler_sq

        phi = np.conj(F_Dh) * rfft2(dh_cur - bh_prev, workers=-1) + np.conj(F_Dv) * rfft2(dv_cur - bv_prev, workers=-1)
        r_numerator = rfft2(s_hat, workers=-1) + c1 * lam * phi

        r_denominator = 1.0 + coeff * Laplacian_response

        r_cur = irfft2(r_numerator / r_denominator, workers=-1)
        # enforce r <= 0
        r_cur = np.minimum(r_cur, 0.0)
        R_cur = np.exp(r_cur)

        # ---------- b update (ADMM dual) ----------
        rh_cur, rv_cur = grad_forward(r_cur)
        bh_cur = bh_prev + R_cur * rh_cur - dh_cur
        bv_cur = bv_prev + R_cur * rv_cur - dv_cur

        

        # ---------- P3: update l via FFT ----------

        L_scaler_sq = np.mean(L_prev) #incorrect in math but adapted for engineering
        l_numerator = rfft2(s - r_cur, workers=-1)
        l_denominator = 1.0 + c2 * L_scaler_sq * Laplacian_response
        l_cur = irfft2(l_numerator / l_denominator, workers=-1).real
        # enforce s <= l (S <= L)
        l_cur = np.maximum(l_cur, s)
        L_cur = np.exp(l_cur)
        
        # ---------- error and value update ----------
        diff_r = np.linalg.norm(r_cur - r_prev) / (np.linalg.norm(r_prev) + 1e-8)
        r_prev = r_cur.copy()
        R_prev = R_cur.copy()
        bh_prev = bh_cur.copy()
        bv_prev = bv_cur.copy()

        diff_l = np.linalg.norm(l_cur - l_prev) / (np.linalg.norm(l_prev) + 1e-8)
        l_prev = l_cur.copy()
        L_prev = L_cur.copy()

        if it % 5 == 0:
            print(f"iter {it}: diff_r={diff_r:.3e}, diff_l={diff_l:.3e}, "
                f"r_min={r_cur.min():.3f}, r_max={r_cur.max():.3f}")

        if iter_count > min_iter and diff_r < eps_1 and diff_l < eps_2:
            break
    
    if iter_count >= max_outer_iter:
        print(iter_count)

    R = np.clip(np.exp(r_prev), 0.0, 1.0)
    L = np.clip(np.exp(l_prev), 0.0, 1.0)
    return R, L


def simple_enhance(img_path: str, out_path: str):
    img = Image.open(img_path).convert("RGB")
    img_np = np.asarray(img).astype(np.float32) / 255.0  # [0,1]
    enhanced = img_np * 3
    enhanced_uint8 = (enhanced * 255.0 + 0.5).astype(np.uint8)
    if out_path is not None:
        Image.fromarray(enhanced_uint8).save(out_path)
    return enhanced_uint8

def simple_gamma(img_path: str, out_path: str, gamma = 2.2):
    img = Image.open(img_path).convert("RGB")
    img_np = np.asarray(img).astype(np.float32) / 255.0  # [0,1]
    img_gamma = np.power(np.clip(img_np, 1e-6, 1.0), 1.0 / gamma)
    enhanced_uint8 = (img_gamma * 255.0 + 0.5).astype(np.uint8)
    if out_path is not None:
        Image.fromarray(enhanced_uint8).save(out_path)
    return enhanced_uint8


def enhance_image(
    img_path: str,
    out_path: str,
    R_path: str,
    L_path: str,
    gamma: float = 2.2
) -> np.ndarray:
    """
    Enhance a low-light RGB image using weighted variational Retinex
    decomposition followed by gamma correction on the illumination.
    Args:
        img_path:
            Path to the input RGB image file.
        out_path:
            Path to save the enhanced RGB image (uint8).
        R_path:
            Path to save the stacked reflectance RGB image (uint8).
        L_path:
            Path to save the stacked illumination RGB image (uint8).
        gamma:
            Gamma correction exponent applied to the illumination component
            (default 2.2). 

    Returns:
        enhanced_uint8:
            Enhanced RGB image in uint8, shape (H, W, 3).
        R_uint8:
            Estimated reflectance RGB image in uint8, shape (H, W, 3).
        L_uint8:
            Estimated illumination RGB image in uint8, shape (H, W, 3).
    """
    img = Image.open(img_path).convert("RGB")
    img_np = np.asarray(img).astype(np.float32) / 255.0  # [0,1]
    h, w, _ = img_np.shape

    R_channels = []
    L_channels = []
    enhanced_channels = []

    for c in range(3):
        S_c = img_np[:, :, c]
        R_c, L_c = decompose_single_channel(
            S_c,
        )
        # Gamma
        L_gamma = np.power(np.clip(L_c, 1e-6, 1.0), 1.0 / gamma)

        S_enh = np.clip(R_c * L_gamma, 0.0, 1.0)

        R_channels.append(R_c)
        L_channels.append(L_c)
        enhanced_channels.append(S_enh)

    R_stack = np.stack(R_channels, axis=-1)
    L_stack = np.stack(L_channels, axis=-1)
    enhanced = np.stack(enhanced_channels, axis=-1)

    enhanced_uint8 = (enhanced * 255.0 + 0.5).astype(np.uint8)
    R_uint8 = (R_stack * 255.0 + 0.5).astype(np.uint8)
    L_uint8 = (L_stack * 255.0 + 0.5).astype(np.uint8)
    if out_path is not None:
        Image.fromarray(enhanced_uint8).save(out_path)
        Image.fromarray(R_uint8).save(R_path)
        Image.fromarray(L_uint8).save(L_path)

    return enhanced_uint8, R_uint8, L_uint8

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python wvm_admm.py input.jpg [output.jpg]")
        sys.exit(0)

    filename = sys.argv[1]
    base_in = "input"
    base_out = "output"
    base_R = "R"
    base_L = "L"
    base_gamma = "gamma"
    base_simple = "simple"

    os.makedirs(base_out, exist_ok=True)
    os.makedirs(base_R, exist_ok=True)
    os.makedirs(base_L, exist_ok=True)
    os.makedirs(base_gamma, exist_ok=True)
    os.makedirs(base_simple, exist_ok=True)

    in_path = os.path.join(base_in, filename)
    out_path = os.path.join(base_out, filename)
    R_path = os.path.join(base_R, filename)
    L_path = os.path.join(base_L, filename)
    simple_path = os.path.join(base_simple, filename)
    gamma_path = os.path.join(base_gamma, filename)


    t_total_start = time.perf_counter()

    t_enh_start = time.perf_counter()
    enhanced_img, R, L = enhance_image(in_path, out_path=out_path, R_path = R_path, L_path = L_path)
    t_enh_end = time.perf_counter()

    t_total_end = time.perf_counter()
    
    simple_enhanced = simple_enhance(in_path, simple_path)
    gamma_enhanced = simple_gamma(in_path, gamma_path)
    enh_time = t_enh_end - t_enh_start
    total_time = t_total_end - t_total_start
    
    print(f"[Timing] enhance_image: {enh_time*1000:.2f} ms")
    print(f"[Timing] total: {total_time*1000:.2f} ms")


    print(f"Saved enhanced result to {out_path}")
