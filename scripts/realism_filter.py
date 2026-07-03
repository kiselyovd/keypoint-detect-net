"""Camera-realism filter for synthetic frames (the 'Unrecord' idea, but data-driven).

Measured gap synth->real CarFusion: real is sharper (x2.6), more saturated (x1.5),
slightly lower contrast (x0.82), and is JPEG (high-freq ringing) while synth is clean
PNG. So the realism stack INCREASES sharpness and saturation and adds JPEG/noise -
the opposite of the naive 'add blur + grain' look.

Stack (geometry-preserving, so keypoints stay valid):
  1. contrast pull-down toward real
  2. saturation boost toward real
  3. unsharp mask (real photos carry more edge energy than soft TAA renders)
  4. fine sensor noise
  5. subtle chromatic aberration (real lens)
  6. mild vignette
  7. JPEG recompression (matches CarFusion's compression artifacts)

apply() is deterministic; apply_random() jitters the parameters per frame for
domain-randomized training data.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def _chromatic_aberration(img, shift=1.2):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    nx, ny = (xx - cx) / cx, (yy - cy) / cy
    b, g, r = cv2.split(img.astype(np.float32))
    mapxr, mapyr = xx + nx * shift, yy + ny * shift
    mapxb, mapyb = xx - nx * shift, yy - ny * shift
    r = cv2.remap(r, mapxr, mapyr, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    b = cv2.remap(b, mapxb, mapyb, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([b, g, r])


def _vignette(img, strength=0.25):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    mask = 1.0 - strength * np.clip(d - 0.4, 0, 1) ** 2
    return img * mask[..., None]


def apply(
    img,
    contrast=0.85,
    saturation=1.45,
    sharp=0.8,
    noise=4.0,
    aberration=1.2,
    vignette=0.22,
    jpeg_q=72,
):
    """Apply the realism stack to a BGR uint8 image; returns BGR uint8."""
    x = img.astype(np.float32)
    # 1. contrast toward real (pull toward mid-gray)
    x = (x - 128.0) * contrast + 128.0
    # 2. saturation boost
    hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    # 3. unsharp mask (add edge energy)
    blur = cv2.GaussianBlur(x, (0, 0), 1.4)
    x = x + sharp * (x - blur)
    # 4. fine sensor noise
    x = x + np.random.normal(0, noise, x.shape).astype(np.float32)
    # 5. chromatic aberration
    x = _chromatic_aberration(np.clip(x, 0, 255), aberration)
    # 6. vignette
    x = _vignette(x, vignette)
    x = np.clip(x, 0, 255).astype(np.uint8)
    # 7. JPEG recompression
    ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_q)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else x


def apply_random(img, rng):
    """Domain-randomized variant: jitter parameters per frame."""
    return apply(
        img,
        contrast=rng.uniform(0.8, 0.95),
        saturation=rng.uniform(1.25, 1.6),
        sharp=rng.uniform(0.5, 1.1),
        noise=rng.uniform(2.0, 6.0),
        aberration=rng.uniform(0.6, 1.8),
        vignette=rng.uniform(0.12, 0.30),
        jpeg_q=int(rng.uniform(60, 85)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    img = cv2.imread(args.inp)
    cv2.imwrite(args.out, apply(img))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
