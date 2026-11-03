"""Projected frontal face, face gallery and face-stabilised close-up of the suspect.

    python enhancement/face_projection.py

Writes (the original video is only read):
    output/face/projected_face_frontal.png   passport-style frontal projection: 13 aligned + fused frames
    output/face/face_pictures.png            one sheet: projected face, views A-C and a strip of single frames
    output/face/face_gallery.png             the face in every second frame 330-398 (single-frame enhanced)
    output/video/suspect_face_closeup.mp4    face-stabilised, 6x zoom, 1/3 speed, frames 330-399

"Projected" here means geometric: the face is shown upright, centred and straight-on, as in a
passport photo. While looking up at the camera (frames 374-386) the suspect faced it almost
directly; the head's roll is about 2 degrees. So the projection needs no rotation or warping,
only alignment, fusion and framing. Only recorded pixels are used. No generative/AI face synthesis.
"""
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import OUT, VIDEO, font, info_bar, read_luma, upright
from fusion import finish, fuse, register
from suspect_images import FACE_VIEWS, GAMMA, USM, grid, labelled, to_img

S = 8
PROJ_FRAMES = range(374, 387)        # suspect looking up at the camera; ECC rho >= 0.97 for all
PROJ_REF, PROJ_CENTER = 381, (535, 185)
TRACK_START, TRACK_RANGE = (381, (535.0, 185.0)), (330, 399)
CLIP_BOX = (80, 96)                  # face close-up window (w, h) in source pixels
CLIP_ZOOM = 6


def _blob_centre(f, c, half=26, pct=93):
    """Centroid of the brightest blob (the IR-lit face) near c."""
    x0, y0 = int(round(c[0])), int(round(c[1]))
    reg = cv2.GaussianBlur(f[y0 - half:y0 + half, x0 - half:x0 + half], (0, 0), 1.5)
    m = (reg > np.percentile(reg, pct)).astype(np.uint8)
    _, lab, st, ce = cv2.connectedComponentsWithStats(m)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    return np.array([ce[k][0] + x0 - half, ce[k][1] + y0 - half])


def track_face(upf):
    """Follow the face from frame 381 backwards and forwards (max 6 px/frame), then smooth."""
    (f0, c0), (first, last) = TRACK_START, TRACK_RANGE
    tr = {f0: np.array(c0)}
    for rng in (range(f0 - 1, first - 1, -1), range(f0 + 1, last + 1)):
        prev = f0
        for f in rng:
            c = _blob_centre(upf[f - 1], tr[prev])
            d = c - tr[prev]
            n = np.linalg.norm(d)
            tr[f] = tr[prev] + (d / n * 6 if n > 6 else d)
            prev = f
    fs = sorted(tr)
    xy = np.array([tr[f] for f in fs])
    k = np.ones(3) / 3
    sm = np.stack([np.convolve(np.pad(xy[:, j], 1, mode="edge"), k, mode="valid") for j in (0, 1)], 1)
    return {f: sm[i] for i, f in enumerate(fs)}


def head_roll(X):
    """Angle (deg) between the face blob's major axis and the vertical."""
    b = cv2.GaussianBlur(X, (0, 0), 6)
    h, w = b.shape
    m = (b > np.percentile(b[h // 4:3 * h // 4, w // 4:3 * w // 4], 70)).astype(np.uint8)
    _, lab, st, _ = cv2.connectedComponentsWithStats(m)
    mo = cv2.moments((lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8), True)
    theta = 0.5 * np.degrees(np.arctan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"]))
    return 90 + theta if theta < 0 else theta - 90, (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"])


def projected_face(up):
    idx = [f - 1 for f in PROJ_FRAMES]
    regs = register(up, PROJ_REF - 1, idx, PROJ_CENTER)
    good = [i for i in idx if regs[i]["rho"] >= 0.97]
    X = fuse(regs, good, S)                        # 640 x 640, reference geometry
    roll, (cx, cy) = head_roll(X)
    # passport framing (4:5): face centred and filling about two thirds of the height
    w, h = 336, 420
    x0, y0 = int(round(cx - w / 2)), int(round(cy - 0.53 * h))
    x0, y0 = max(0, min(x0, X.shape[1] - w)), max(0, min(y0, X.shape[0] - h))
    Xc = X[y0:y0 + h, x0:x0 + w]
    face = finish(Xc, np.percentile(Xc, 2), Xc.max() * 1.02, S, gamma=GAMMA, usm=USM).astype(np.float32)
    # gentle "clarity" (large-radius unsharp mask, 2.5 source pixels) so the eye sockets,
    # nose and mouth shading read more easily; it only re-weights detail that is present
    face = np.clip(face + 0.35 * (face - cv2.GaussianBlur(face, (0, 0), 2.5 * S)) + 0.5, 0, 255).astype(np.uint8)
    face = cv2.resize(face, (600, 750), interpolation=cv2.INTER_LANCZOS4)
    to_img(face).save(OUT / "face" / "projected_face_frontal.png")
    rhos = [regs[i]["rho"] for i in good]
    return face, dict(n=len(good), roll=roll, rho_min=min(rhos))


def _face_crop(f, c, w, h):
    return cv2.getRectSubPix(f, (w, h), (float(c[0]), float(c[1]) + 4))


def single_face(upf, tr, fno, w=64, h=80, s=3):
    """Single-frame face: 3-frame NLM on co-located crops, bicubic x s, auto levels."""
    stack = [_face_crop(upf[j - 1], tr[fno], w, h) for j in (fno - 1, fno, fno + 1)]
    st8 = [np.clip(c * 4, 0, 255).astype(np.uint8) for c in stack]   # spread levels for NLM
    d = cv2.fastNlMeansDenoisingMulti(st8, 1, 3, None, 6, 5, 11).astype(np.float32) / 4
    up_ = cv2.resize(d, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return finish(up_, np.percentile(up_, 2), max(up_.max() * 1.02, np.percentile(up_, 2) + 6), s,
                  gamma=GAMMA, usm=0.4)


def gallery(upf, tr):
    tiles = [labelled(to_img(single_face(upf, tr, f)), f"Frame {f}", size=14)
             for f in range(330, 399, 2)]
    g = grid(tiles, 7, gap=8)
    g.save(OUT / "face" / "face_gallery.png")
    return tiles


def closeup_video(raw, upf, tr):
    first, last = TRACK_RANGE
    w, h = CLIP_BOX
    W, H = w * CLIP_ZOOM, h * CLIP_ZOOM
    bar_h = 40
    n_frames = len(upf)
    los, his = [], []
    for f in range(first, last + 1):
        c = _face_crop(upf[f - 1], tr[f], w, h)
        los.append(np.percentile(c, 2))
        his.append(np.percentile(c, 99.9))
    k = np.ones(5) / 5
    los = np.convolve(np.pad(los, 2, mode="edge"), k, mode="valid")
    his = np.convolve(np.pad(his, 2, mode="edge"), k, mode="valid")
    num, den = 25170, 1679 * 3                     # one third of the source frame rate
    path = OUT / "video" / "suspect_face_closeup.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                            "-s", f"{W}x{H + bar_h}", "-r", f"{num}/{den}", "-i", "-",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    for n, f in enumerate(range(first, last + 1)):
        # neighbours are cut at this frame's face position (stabilised window) for temporal NLM
        nb = [min(max(j, 1), n_frames) for j in (f - 1, f, f + 1)]
        st8 = [np.clip(_face_crop(upf[j - 1], tr[f], w, h) * 4, 0, 255).astype(np.uint8) for j in nb]
        d = cv2.fastNlMeansDenoisingMulti(st8, 1, 3, None, 6, 5, 11).astype(np.float32) / 4
        z = cv2.resize(d, (W, H), interpolation=cv2.INTER_LANCZOS4)
        y = finish(z, los[n], max(his[n] * 1.02, los[n] + 6), CLIP_ZOOM, gamma=GAMMA, usm=0.4)
        bar = info_bar(W, raw[f - 1], f"Frame {f}", bar_h)
        enc.stdin.write(np.vstack([bar, y]).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit("ffmpeg failed")


def sheet(proj, info, tiles):
    views = [labelled(Image.open(OUT / "face" / v["file"]).resize((300, 360), Image.LANCZOS), v["title"],
                      f"frames {v['frames'][0]}–{v['frames'][-1]} · {v['clock']}", size=15)
             for v in FACE_VIEWS]
    big = labelled(to_img(proj).resize((480, 600), Image.LANCZOS), "Projected frontal face",
                   f"{info['n']} aligned frames {PROJ_FRAMES[0]}–{PROJ_FRAMES[-1]} · camera clock "
                   "01:26:42–43", size=20)
    right = grid(views, 3, gap=10)
    strip = grid([t.resize((int(t.width * 0.8), int(t.height * 0.8)), Image.LANCZOS)
                  for t in tiles[13:33:2]], 10, gap=6)          # frames 356, 360, ... 392
    width = max(big.width + right.width + 30, strip.width + 20)
    head, top_h = 96, max(big.height, right.height)
    H = head + top_h + 46 + strip.height + 16
    board = Image.new("L", (width, H), 14)
    d = ImageDraw.Draw(board)
    d.text((20, 18), "Suspect's face — enhanced pictures and projected frontal face", fill=245,
           font=font(30, bold=True))
    d.text((20, 60), "thief_video.mp4 · IMOU night-vision camera · 2026-10-31, camera clock "
                     "01:26:41–01:26:44 · suspect at the foot of the house steps", fill=175, font=font(16))
    board.paste(big, (10, head))
    rx = big.width + 20
    board.paste(right, (rx, head))
    notes = [
        ("How these pictures were made", 235, True),
        ("Only from pixels the camera recorded: frame alignment, averaging,", 190, False),
        ("enlargement and brightness/contrast. No AI-generated detail.", 190, False),
        ("", 0, False),
        ("Projected frontal face: the suspect looked almost straight at the", 165, False),
        (f"camera (head roll about {abs(info['roll']):.0f}°), so no rotation or warping was needed:", 165, False),
        (f"{info['n']} frames were aligned to a fraction of a pixel, averaged and framed", 165, False),
        ("passport-style. Finer detail than this was never recorded.", 165, False),
        ("", 0, False),
        ("Infrared night vision: grey tones do not show real skin or clothing colours.", 140, False),
    ]
    y = head + right.height + 14
    for text, shade, bold in notes:
        if text:
            d.text((rx + 12, y), text, fill=shade, font=font(17 if bold else 15, bold=bold))
        y += 30 if bold else 24
    d.text((20, head + top_h + 12), "Single frames (each enhanced on its own)", fill=235, font=font(18, bold=True))
    board.paste(strip, (10, head + top_h + 40))
    board.save(OUT / "face" / "face_pictures.png")


def main():
    raw, _ = read_luma(VIDEO)
    up = upright(raw)
    upf = up.astype(np.float32)
    (OUT / "face").mkdir(parents=True, exist_ok=True)
    (OUT / "video").mkdir(parents=True, exist_ok=True)
    proj, info = projected_face(up)
    print(f"projected face: {info['n']} frames fused, min ECC rho {info['rho_min']:.3f}, "
          f"head roll {info['roll']:+.1f} deg")
    tr = track_face(upf)
    tiles = gallery(upf, tr)
    closeup_video(raw, upf, tr)
    sheet(proj, info, tiles)
    print("written: projected_face_frontal.png, face_gallery.png, face_pictures.png, suspect_face_closeup.mp4")


if __name__ == "__main__":
    main()
