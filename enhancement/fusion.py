"""Multi-frame fusion of the suspect's face.

A single CCTV frame of the face is ~27 x 33 pixels, crushed into ~45 grey levels
and carrying H.264 compression noise. When the head is (nearly) still for a
fraction of a second, the same face is recorded several times with different
noise and slightly different sub-pixel positions. Registering those frames to
each other and averaging them reduces noise / compression artefacts without
inventing anything - every output value is an average of recorded pixels.

Steps
1. coarse integer alignment by normalised cross-correlation of the face patch;
2. sub-pixel *affine* alignment with ECC (Evangelidis & Psarakis, 2008)
   evaluated only inside an elliptical face mask, so the static background
   cannot pull the estimate towards "no motion";
3. each frame is resampled once (bicubic) onto an s-times finer grid in the
   reference frame's geometry;
4. robust per-pixel average (trimmed mean: per-pixel min and max dropped) to
   reject residual mis-registration and outliers.
"""
import cv2
import numpy as np

ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)


def _ecc_affine(T, crop, mask):
    """Affine ECC of `crop` against template T, evaluated on the template mask.
    Returns (rho, W) with W mapping template coordinates to crop coordinates."""
    W = np.eye(2, 3, dtype=np.float32)
    if hasattr(cv2, "findTransformECCWithMask"):  # recent OpenCV
        return cv2.findTransformECCWithMask(T, crop, mask, np.full_like(mask, 255), W,
                                            cv2.MOTION_AFFINE, ECC_CRITERIA, 3)
    # Older OpenCV only accepts a mask for the image being warped, so solve the inverse
    # problem (warp the masked reference onto the crop) and invert the result.
    rho, Winv = cv2.findTransformECC(crop, T, W, cv2.MOTION_AFFINE, ECC_CRITERIA, mask, 3)
    return rho, cv2.invertAffineTransform(Winv)


def register(frames, ref, idx, center, R=40, axes=(21, 26), search=30):
    """Align the face in frames[i] (i in idx, 0-based) to frames[ref].

    center: (x, y) of the face in the reference frame (upright coordinates).
    Returns {i: dict(rho, ncc, W, crop, shift)}: ECC correlation, coarse NCC score,
    the 2x3 affine W mapping reference-crop coordinates to coordinates inside `crop`
    (the 2R x 2R crop of frame i), and the integer shift used to cut that crop.
    """
    fx, fy = int(round(center[0])), int(round(center[1]))
    refimg = frames[ref].astype(np.float32)
    T = refimg[fy - R:fy + R, fx - R:fx + R]
    mask = np.zeros(T.shape, np.uint8)
    cv2.ellipse(mask, (R, R), axes, 0, 0, 360, 255, -1)
    tmpl = refimg[fy - axes[1]:fy + axes[1], fx - axes[0]:fx + axes[0]]
    out = {}
    for i in idx:
        f = frames[i].astype(np.float32)
        S = f[fy - axes[1] - search:fy + axes[1] + search, fx - axes[0] - search:fx + axes[0] + search]
        _, ncc, _, loc = cv2.minMaxLoc(cv2.matchTemplate(S, tmpl, cv2.TM_CCOEFF_NORMED))
        dx, dy = loc[0] - search, loc[1] - search
        crop = f[fy - R + dy:fy + R + dy, fx - R + dx:fx + R + dx].copy()
        try:
            rho, W = _ecc_affine(T, crop, mask)
        except cv2.error:  # ECC did not converge: exclude the frame from fusion
            rho, W = 0.0, np.eye(2, 3, dtype=np.float32)
        out[i] = dict(rho=float(rho), ncc=float(ncc), W=W.astype(np.float64), crop=crop, shift=(dx, dy))
    return out


def hr_matrix(W, s):
    """Matrix mapping fine-grid (s x) reference coordinates to coordinates in the frame crop."""
    A, b = W[:, :2], W[:, 2]
    c = 0.5 / s - 0.5  # pixel-centre convention
    return np.hstack([A / s, (A @ np.array([c, c]) + b)[:, None]])


def aligned_stack(regs, idx, s):
    out = []
    for i in idx:
        crop = regs[i]["crop"]
        n = crop.shape[0] * s
        out.append(cv2.warpAffine(crop, hr_matrix(regs[i]["W"], s), (n, n),
                                  flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REFLECT))
    return np.stack(out)


def fuse(regs, idx, s):
    """Trimmed-mean fusion of the registered frames on an s-times finer grid."""
    al = aligned_stack(regs, idx, s)
    if len(idx) >= 4:
        al = np.sort(al, axis=0)[1:-1]
    return al.mean(axis=0)


def finish(X, lo, hi, s, gamma=0.9, usm=0.5):
    """Levels + gamma + gentle unsharp mask (applied identically to every panel)."""
    y = np.clip((X - lo) / (hi - lo), 0, 1) ** gamma * 255.0
    if usm > 0:
        y = y + usm * (y - cv2.GaussianBlur(y, (0, 0), 0.8 * s))
    return np.clip(y + 0.5, 0, 255).astype(np.uint8)
