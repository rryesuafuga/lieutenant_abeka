"""Shared helpers for the CCTV enhancement pipeline.

Everything here is classical, deterministic image processing (no generative /
"AI face restoration" models), so every output pixel can be traced back to
pixels recorded by the camera.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
VIDEO = REPO / "thief_video.mp4"
OUT = REPO / "output"

# Burned-in camera clock ("2026-10-31 01:26:43") in the *unrotated* frame: y0, y1, x0, x1
CLOCK_BOX = (20, 50, 738, 1000)

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
_FONT_BOLD_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def font(size, bold=False):
    for p in _FONT_BOLD_PATHS if bold else _FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path=VIDEO):
    """Frame size, average frame rate and per-frame timestamps / picture types."""
    s = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout)["streams"][0]
    num, den = map(int, s["avg_frame_rate"].split("/"))
    fr = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "frame=pts_time,pict_type,pkt_size", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout)["frames"]
    return dict(width=int(s["width"]), height=int(s["height"]), fps=f"{num}/{den}",
                times=[float(f["pts_time"]) for f in fr],
                types=[f["pict_type"] for f in fr],
                sizes=[int(f["pkt_size"]) for f in fr])


def read_luma(path=VIDEO):
    """Decode the luma (Y) plane of every frame, without dropping or duplicating frames.

    The footage is monochrome night-vision (chroma is flat), so luma is the whole signal.
    """
    info = probe(path)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, info["height"], info["width"])
    return frames, info


def upright(frames):
    """The camera is mounted upside down (e.g. under an eave) without image flip
    enabled, so the recorded picture is rotated 180 degrees. Undo that."""
    return np.ascontiguousarray(frames[..., ::-1, ::-1])


def gamma_schedule(frames_up, target=0.33, g_min=0.45, g_max=1.0, win=15):
    """Exposure-adaptive gamma per frame, smoothed over ~1 s so it does not flicker.

    Around the house steps the camera's auto-exposure darkens the whole picture
    (median ~19/255) because of the bright, IR-lit wall; elsewhere the median is
    ~70-95. A per-frame gamma brings the scene median towards `target`.
    """
    med = np.array([np.median(f) for f in frames_up], np.float64) / 255.0
    med = np.clip(med, 1 / 255, 0.999)
    k = np.ones(win) / win
    sm = np.convolve(np.pad(np.log(med), win // 2, mode="edge"), k, mode="valid")
    return np.clip(np.log(target) / sm, g_min, g_max)


def tone(img_u8, g):
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.LUT(img_u8, lut)


def enhance_frame(frames_up, i, g, t_win=5):
    """Enhance one upright frame using its temporal neighbours.

    1. gamma lift (applied *before* denoising so that the temporal average can
       represent in-between grey levels -> no banding in the lifted shadows)
    2. spatio-temporal non-local-means denoising over `t_win` frames
       (strength scales with the gamma slope, i.e. with how much noise was lifted)
    3. mild CLAHE (local contrast) + mild unsharp mask
    """
    n = len(frames_up)
    half = t_win // 2
    idx = [min(max(j, 0), n - 1) for j in range(i - half, i + half + 1)]
    stack = [tone(frames_up[j], g) for j in idx]
    h = 2.5 + 4.0 * (1.0 - g)
    den = cv2.fastNlMeansDenoisingMulti(stack, half, t_win, None, h, 7, 21)
    y = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(den).astype(np.float32)
    y += 0.5 * (y - cv2.GaussianBlur(y, (0, 0), 1.2))
    return np.clip(y + 0.5, 0, 255).astype(np.uint8)


def clock_patch(frame_raw):
    """Original, unmodified pixels of the burned-in camera clock (readable orientation)."""
    y0, y1, x0, x1 = CLOCK_BOX
    return frame_raw[y0:y1, x0:x1]


def info_bar(width, frame_raw, left_text, height=40):
    """Dark caption bar: free text on the left, the camera's own clock pixels on the right."""
    bar = Image.new("L", (width, height), 14)
    d = ImageDraw.Draw(bar)
    d.text((10, height // 2), left_text, fill=225, font=font(15), anchor="lm")
    p = clock_patch(frame_raw)
    ph, pw = p.shape
    x = width - pw - 8
    d.text((x - 8, height // 2), "camera clock:", fill=150, font=font(13), anchor="rm")
    bar.paste(Image.fromarray(np.ascontiguousarray(p)), (x, (height - ph) // 2))
    return np.asarray(bar)
