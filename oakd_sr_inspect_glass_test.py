import cv2
import numpy as np
import argparse
import time

"""
Test harness for Glass Glue inspection algorithms (no DepthAI required).

Usage:
  python oakd_sr_inspect_glass_test.py            # synthetic test images
  python oakd_sr_inspect_glass_test.py --mono img.png --depth depth.png
  python oakd_sr_inspect_glass_test.py --webcam

This script duplicates the processing (CLAHE, glue-mask, depth viz)
so you can tune parameters before deploying to OAK-D.
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mono', help='Path to mono (grayscale) image')
    p.add_argument('--depth', help='Path to depth image (single channel)')
    p.add_argument('--webcam', action='store_true', help='Use webcam for mono feed')
    return p.parse_args()


clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def compute_glue_mask_from_mono(mono_img):
    enhanced = clahe.apply(mono_img)
    large_blur = cv2.GaussianBlur(enhanced, (31, 31), 0)
    high = cv2.subtract(enhanced, large_blur)
    high = cv2.normalize(high, None, 0, 255, cv2.NORM_MINMAX)
    lap = cv2.Laplacian(enhanced, cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    combined = cv2.addWeighted(high, 0.8, lap, 0.4, 0)
    _, th = cv2.threshold(combined, 18, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return closed


def synth_mono(size=(640, 480)):
    w, h = size
    # gradient background
    x = np.linspace(0, 255, w, dtype=np.uint8)
    grad = np.tile(x, (h, 1))
    # add a few darker glue-like bands
    img = grad.copy()
    for y in range(120, 360, 60):
        cv2.rectangle(img, (80, y), (w-80, y+8), 60, -1)
        cv2.GaussianBlur(img, (9, 9), 0, dst=img)
    return img


def synth_depth(size=(640, 480)):
    w, h = size
    xv, yv = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    depth = (1 - np.sqrt((xv - 0.5)**2 + (yv - 0.5)**2)) * 2000
    depth = np.clip(depth, 0, 2000).astype(np.float32)
    # add a small bump to simulate an object
    cv2.circle(depth, (w//2, h//2), 60, 800, -1)
    return depth


def normalize_depth_to_u8(depth):
    d = depth.copy()
    valid = d > 0
    if valid.any():
        lo = np.percentile(d[valid], 1)
        hi = np.percentile(d[valid], 99)
        norm = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
        u8 = (norm * 255).astype(np.uint8)
    else:
        u8 = np.zeros(depth.shape, dtype=np.uint8)
    return u8


def main():
    args = parse_args()

    if args.webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('Webcam not available, falling back to synthetic')
            cap = None
    else:
        cap = None

    if args.mono:
        mono_img = cv2.imread(args.mono, cv2.IMREAD_GRAYSCALE)
        if mono_img is None:
            print('Could not read mono image, using synthetic')
            mono_img = synth_mono()
    else:
        mono_img = synth_mono()

    if args.depth:
        depth_img = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            print('Could not read depth image, using synthetic')
            depth_img = synth_depth()
        else:
            if depth_img.ndim == 3:
                depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                depth_img = depth_img.astype(np.float32)
    else:
        depth_img = synth_depth()

    mode = 0
    colormap_idx = 0
    COLORMAPS = [cv2.COLORMAP_TURBO, cv2.COLORMAP_JET, cv2.COLORMAP_MAGMA]
    screenshot_idx = 0

    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                break
            mono = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            mono = mono_img

        # compute mask
        mask = compute_glue_mask_from_mono(mono)

        # depth visualization
        u8 = normalize_depth_to_u8(depth_img)
        depth_vis = cv2.applyColorMap(255 - u8, COLORMAPS[colormap_idx])

        mono_clahe = clahe.apply(mono)
        mono_bgr = cv2.cvtColor(mono_clahe, cv2.COLOR_GRAY2BGR)

        if mode == 0:
            overlay = mono_bgr.copy()
            overlay[mask > 0] = (0, 0, 255)
            out = cv2.addWeighted(overlay, 0.45, mono_bgr, 0.55, 0)
            combined = np.hstack((cv2.resize(out, (640, 360)), cv2.resize(depth_vis, (640, 360))))
        elif mode == 1:
            edges = cv2.Canny(mono_clahe, 50, 150)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((cv2.resize(mono_bgr, (640, 360)), cv2.resize(edges_bgr, (640, 360))))
        else:
            combined = np.hstack((cv2.resize(mono_bgr, (640, 360)), cv2.resize(depth_vis, (640, 360))))

        cv2.putText(combined, f'Mode: {mode}  Colormap: {colormap_idx}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(combined, 'Q=quit S=screenshot M=mode C=colormap', (10, combined.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow('Glass Glue Inspect - TEST', combined)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('m'):
            mode = (mode + 1) % 3
        elif k == ord('c'):
            colormap_idx = (colormap_idx + 1) % len(COLORMAPS)
        elif k == ord('s'):
            fname = f'test_screenshot_{int(time.time())}_{screenshot_idx}.png'
            cv2.imwrite(fname, combined)
            print('Saved', fname)
            screenshot_idx += 1

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
