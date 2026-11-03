"""Produce an orientation-corrected, enhanced copy of the whole CCTV clip, plus a
zoomed, half-speed close-up of the suspect at the house steps.

    python enhancement/enhance_video.py

Outputs (the original thief_video.mp4 is only read, never modified):
    output/video/thief_video_enhanced.mp4      full clip, upright, enhanced, original audio
    output/video/suspect_closeup_slowmo.mp4    frames 245-410, 2x zoom, half speed
"""
import subprocess
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

from common import OUT, VIDEO, enhance_frame, gamma_schedule, info_bar, read_luma, upright

BAR_H = 40
# Close-up of the suspect at the house steps (1-based frame numbers, upright coordinates).
CLOSE_FRAMES = (245, 410)
CLOSE_BOX = (90, 450, 400, 720)  # y0, y1, x0, x1
CLOSE_ZOOM = 2

_UP = _G = None  # shared with worker processes (inherited via fork)


def _work(i):
    return enhance_frame(_UP, i, _G[i])


def _encoder(path, w, h, fps, audio_from=None, crf=18):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
           "-s", f"{w}x{h}", "-r", fps, "-i", "-"]
    if audio_from:
        # no -shortest: the audio track is ~0.1 s shorter than the video and every frame must be kept
        cmd += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a?", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    global _UP, _G
    t0 = time.time()
    raw, info = read_luma(VIDEO)
    n, H, W = raw.shape
    print(f"decoded {n} frames {W}x{H} @ {info['fps']} fps", flush=True)
    _UP = upright(raw)
    _G = gamma_schedule(_UP)

    vdir = OUT / "video"
    vdir.mkdir(parents=True, exist_ok=True)
    full = _encoder(vdir / "thief_video_enhanced.mp4", W, H + BAR_H, info["fps"], audio_from=VIDEO)

    y0, y1, x0, x1 = CLOSE_BOX
    cw, ch = (x1 - x0) * CLOSE_ZOOM, (y1 - y0) * CLOSE_ZOOM
    num, den = map(int, info["fps"].split("/"))
    close = _encoder(vdir / "suspect_closeup_slowmo.mp4", cw, ch + BAR_H, f"{num}/{den * 2}", crf=16)

    with Pool(4) as pool:
        for i, enh in enumerate(pool.imap(_work, range(n), chunksize=4)):
            fno = i + 1
            text = f"Frame {fno}/{n}   t = {info['times'][i]:6.2f} s   (rotated 180°, enhanced)"
            full.stdin.write(np.vstack([info_bar(W, raw[i], text, BAR_H), enh]).tobytes())
            if CLOSE_FRAMES[0] <= fno <= CLOSE_FRAMES[1]:
                crop = cv2.resize(enh[y0:y1, x0:x1], (cw, ch), interpolation=cv2.INTER_LANCZOS4)
                bar = info_bar(cw, raw[i], f"Frame {fno}  ·  2x zoom, ½ speed", BAR_H)
                close.stdin.write(np.vstack([bar, crop]).tobytes())
            if fno % 100 == 0:
                print(f"  {fno}/{n} frames  ({time.time() - t0:.0f} s)", flush=True)
    for p in (full, close):
        p.stdin.close()
        if p.wait() != 0:
            sys.exit("ffmpeg failed")
    print(f"done in {time.time() - t0:.0f} s -> {vdir}")


if __name__ == "__main__":
    main()
