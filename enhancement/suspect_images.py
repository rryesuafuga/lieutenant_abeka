"""Enhanced still images of the suspect from thief_video.mp4.

    python enhancement/suspect_images.py

Writes (the original video is only read):
    output/face/face_A_looking_up.png         multi-frame fused face, frames 376-384
    output/face/face_B_three_quarter.png      multi-frame fused face, frames 385-389
    output/face/face_C_head_lowered.png       multi-frame fused face, frames 356-360
    output/face/face_enhancement_steps.png    as recorded -> levels -> single frame -> fused
    output/stills/frame_NNNN.png              full frame, upright + enhanced, with caption bar
    output/stills/suspect_frame_NNNN.png      zoomed crop of the suspect
    output/stills/shoes_taken_sequence.png    frames 393-399 at the steps
    output/suspect_summary.png                one-page overview
    output/registration_log.txt               alignment quality for every fused frame
Frame numbers are 1-based (frame 1 = first frame of the file).
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import OUT, VIDEO, enhance_frame, font, gamma_schedule, info_bar, read_luma, sha256, upright
from fusion import finish, fuse, register

S = 8                  # face enlargement factor (fusion grid)
GAMMA, USM = 1.1, 0.6  # face tone curve / unsharp-mask amount (identical for every comparison panel)
MIN_RHO = 0.97         # frames whose ECC alignment correlation is lower are not fused

FACE_VIEWS = [
    dict(key="A", file="face_A_looking_up.png", title="A · looking up at the camera",
         ref=381, frames=range(376, 385), center=(535, 185), clock="01:26:42–43"),
    dict(key="B", file="face_B_three_quarter.png", title="B · head turned (three-quarter)",
         ref=387, frames=range(385, 390), center=(534, 188), clock="01:26:43"),
    dict(key="C", file="face_C_head_lowered.png", title="C · head slightly lowered",
         ref=357, frames=range(356, 361), center=(585, 187), clock="01:26:41"),
]
FACE_WIN = (-30, 30, -34, 38)  # x0, x1, y0, y1 around the face centre (source pixels)

# Key moments (camera clock read from the burned-in overlay of each frame).
# box = suspect crop in upright coordinates: y0, y1, x0, x1
KEY_FRAMES = {
    13: dict(clock="01:26:18", desc="walking through the yard", short="walking through the yard",
             box=(400, 545, 165, 310)),
    94: dict(clock="01:26:24", desc="the camera follows them across the yard", short="followed across the yard",
             box=(285, 455, 540, 700)),
    206: dict(clock="01:26:31", desc="approaching the house steps", short="approaching the steps",
              box=(105, 295, 505, 675)),
    257: dict(clock="01:26:34", desc="bending over at the house steps", short="Bending over at the steps",
              box=(120, 420, 420, 680)),
    345: dict(clock="01:26:40", desc="crouching at the steps, facing the camera", short="crouching at the steps",
              box=(120, 420, 450, 710)),
    381: dict(clock="01:26:43", desc="standing at the steps, looking up at the camera",
              short="At the steps, looking up", box=(130, 430, 440, 700)),
    398: dict(clock="01:26:44", desc="picks up the pair of shoes from the step", short="Picks up the shoes",
              box=(130, 430, 440, 700)),
}
SHOE_FRAMES = [393, 396, 397, 398, 399]  # all at camera clock 01:26:44
SHOE_BOX = (250, 420, 490, 690)


def to_img(a):
    return Image.fromarray(np.ascontiguousarray(a))


def labelled(img, title, sub=None, width=None, pad=10, bg=14, size=18):
    """Stack a caption under a PIL image."""
    width = width or img.width
    h = size + 12 + (22 if sub else 0) + pad
    out = Image.new("L", (width, img.height + h), bg)
    out.paste(img, ((width - img.width) // 2, 0))
    d = ImageDraw.Draw(out)
    d.text((pad, img.height + 6), title, fill=235, font=font(size, bold=True))
    if sub:
        d.text((pad, img.height + size + 14), sub, fill=170, font=font(15))
    return out


def grid(images, cols, gap=12, bg=14):
    rows = [images[i:i + cols] for i in range(0, len(images), cols)]
    cw = max(im.width for im in images)
    rh = [max(im.height for im in r) for r in rows]
    out = Image.new("L", (cols * cw + (cols + 1) * gap, sum(rh) + (len(rows) + 1) * gap), bg)
    y = gap
    for r, h in zip(rows, rh):
        for j, im in enumerate(r):
            out.paste(im, (gap + j * (cw + gap), y))
        y += h + gap
    return out


def face_images(up, log):
    (OUT / "face").mkdir(parents=True, exist_ok=True)
    x0, x1, y0, y1 = FACE_WIN
    results, steps = [], []
    for v in FACE_VIEWS:
        idx = [f - 1 for f in v["frames"]]
        regs = register(up, v["ref"] - 1, idx, v["center"])
        log.append(f"View {v['key']}: reference frame {v['ref']}, face centre {v['center']}")
        for i in idx:
            r, W = regs[i], regs[i]["W"]
            log.append(f"  frame {i + 1}: NCC {r['ncc']:.3f}  ECC rho {r['rho']:.4f}  integer shift {r['shift']}"
                       f"  affine t=({W[0, 2]:+.2f},{W[1, 2]:+.2f}) A=[{W[0, 0]:.3f} {W[0, 1]:+.3f};"
                       f" {W[1, 0]:+.3f} {W[1, 1]:.3f}]")
        good = [i for i in idx if regs[i]["rho"] >= MIN_RHO]
        log.append(f"  fused {len(good)}/{len(idx)} frames (ECC rho >= {MIN_RHO})")
        X = fuse(regs, good, S)
        R = regs[idx[0]]["crop"].shape[0] // 2
        sl = (slice((R + y0) * S, (R + y1) * S), slice((R + x0) * S, (R + x1) * S))
        Xc = X[sl]
        # black point at the dark background, white point just above the brightest fused
        # value (no clipping, so skin highlights keep their detail)
        lo, hi = np.percentile(Xc, 2), Xc.max() * 1.02
        final = finish(Xc, lo, hi, S, gamma=GAMMA, usm=USM)
        to_img(final).save(OUT / "face" / v["file"])
        results.append((v, final))

        # progression panels, all with the same black/white points, for transparency
        ref_crop = regs[v["ref"] - 1]["crop"]
        nn = np.kron(ref_crop, np.ones((S, S), np.float32))[sl]
        bic = cv2.resize(ref_crop, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)[sl]
        steps += [
            labelled(to_img(np.clip(nn, 0, 255).astype(np.uint8)), "As recorded", f"frame {v['ref']}, pixels enlarged"),
            labelled(to_img(finish(nn, lo, hi, S, gamma=GAMMA, usm=0)), "Levels only",
                     f"frame {v['ref']}, brightness/contrast"),
            labelled(to_img(finish(bic, lo, hi, S, gamma=GAMMA, usm=USM)), "Single frame enhanced",
                     f"frame {v['ref']}, interpolated + sharpened"),
            labelled(to_img(final), f"Multi-frame fused ({v['key']})",
                     f"{len(good)} frames {v['frames'][0]}–{v['frames'][-1]} aligned + averaged"),
        ]
    grid(steps, 4).save(OUT / "face" / "face_enhancement_steps.png")
    return results


def stills(raw, up, g):
    (OUT / "stills").mkdir(parents=True, exist_ok=True)
    n, H, W = up.shape
    enh = {f: enhance_frame(up, f - 1, g[f - 1]) for f in sorted(set(KEY_FRAMES) | set(SHOE_FRAMES))}
    crops = {}
    for f, k in KEY_FRAMES.items():
        bar = info_bar(W, raw[f - 1], f"Frame {f}/{n}  ·  {k['desc']}")
        to_img(np.vstack([bar, enh[f]])).save(OUT / "stills" / f"frame_{f:04d}.png")
        y0, y1, x0, x1 = k["box"]
        crops[f] = cv2.resize(enh[f][y0:y1, x0:x1], None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LANCZOS4)
        to_img(crops[f]).save(OUT / "stills" / f"suspect_frame_{f:04d}.png")
    y0, y1, x0, x1 = SHOE_BOX
    seq = [labelled(to_img(cv2.resize(enh[f][y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)),
                    f"Frame {f}", "camera clock 01:26:44") for f in SHOE_FRAMES]
    grid(seq, len(seq)).save(OUT / "stills" / "shoes_taken_sequence.png")
    return enh, crops


def summary(faces, enh, crops, digest):
    TW, FH = 440, 528  # tile width, face tile height
    tiles = []
    for v, img in faces:
        tiles.append(labelled(to_img(img).resize((TW, FH), Image.LANCZOS), v["title"],
                              f"frames {v['frames'][0]}–{v['frames'][-1]} · camera clock {v['clock']}"))
    for f in (381, 257, 398):
        k = KEY_FRAMES[f]
        im = to_img(crops[f])
        im = im.resize((TW, int(im.height * TW / im.width)), Image.LANCZOS)
        if im.height > FH:
            top = (im.height - FH) // 2
            im = im.crop((0, top, TW, top + FH))
        tiles.append(labelled(im, k["short"], f"frame {f} · camera clock {k['clock']}"))
    top_grid = grid(tiles, 3)

    tl = [labelled(to_img(enh[f]).resize((256, 144), Image.LANCZOS), KEY_FRAMES[f]["clock"],
                   KEY_FRAMES[f]["short"].lower() if f in (381, 398) else KEY_FRAMES[f]["short"], size=16)
          for f in (13, 94, 206, 381, 398)]
    tl_grid = grid(tl, 5)

    lines = [
        ("Appearance: dark long-sleeved top (appears hooded, hood down) with a small bright mark on the front "
         "(wearer's left, waist height);", 205),
        ("light-toned trousers, short-cropped hair; footwear tone varies between frames. Infrared night "
         "vision: tones may not match the real colours.", 205),
        ("Faces A–C: consecutive frames aligned to sub-pixel accuracy and averaged to reduce noise and "
         "compression artefacts.", 160),
        ("No AI / generative face reconstruction was used — nothing was added that the camera did not "
         "record.", 160),
        (f"Source SHA-256 {digest} (original left unmodified) · frame numbers are 1-based.", 130),
    ]
    width = max(top_grid.width, tl_grid.width)
    head_h, foot_h = 112, 26 * len(lines) + 30
    board = Image.new("L", (width, head_h + top_grid.height + 48 + tl_grid.height + foot_h), 14)
    d = ImageDraw.Draw(board)
    d.text((22, 20), "Suspect — enhanced stills from thief_video.mp4", fill=245, font=font(32, bold=True))
    d.text((22, 66), "IMOU night-vision camera · camera clock 2026-10-31 01:26:17 – 01:27:13 · "
                     "recording was upside-down; rotated 180°", fill=175, font=font(16))
    board.paste(top_grid, ((width - top_grid.width) // 2, head_h))
    y = head_h + top_grid.height + 6
    d.text((22, y + 6), "Timeline (camera clock)", fill=235, font=font(20, bold=True))
    board.paste(tl_grid, ((width - tl_grid.width) // 2, y + 36))
    y += 36 + tl_grid.height + 10
    for k, (line, shade) in enumerate(lines):
        d.text((22, y + k * 26), line, fill=shade, font=font(15))
    board.save(OUT / "suspect_summary.png")


def main():
    raw, info = read_luma(VIDEO)
    up = upright(raw)
    g = gamma_schedule(up)
    digest = sha256(VIDEO)
    log = [f"Source: {VIDEO.name}  SHA-256 {digest}  frames {len(raw)}"]
    faces = face_images(up, log)
    enh, crops = stills(raw, up, g)
    summary(faces, enh, crops, digest)
    (OUT / "registration_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
