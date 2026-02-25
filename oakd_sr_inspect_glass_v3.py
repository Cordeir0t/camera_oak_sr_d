import cv2
import depthai as dai
import numpy as np
import os

"""
OAK-D SR — Inspeção de faixas de cola em vidro (mono only)

Baseado em `oakd_sr_inspect_rodas.py` mas adaptado para vidro.
Modos: 0 = Mask+Depth, 1 = Edges, 2 = Depth only
Modo edges é mais para debug, pode não ser útil para inspeção real. O modo principal é o MASK+DEPTH que sobrepõe a máscara de detecção de cola no feed mono, ao lado da visualização de profundidade. 
modo edges pode ajudar a ajustar os parâmetros de detecção de bordas e contraste, mas tem mais ruído e pode não ser tão claro para operadores humanos.
modo depth only é para avaliar a qualidade do mapa de profundidade sem distrações.
Teclas: Q = sair | C = colormap | M = modo | S = screenshot
"""

COLORMAPS = [cv2.COLORMAP_TURBO, cv2.COLORMAP_JET, cv2.COLORMAP_MAGMA]
colormap_idx = 0
MODES = ["MASK+DEPTH", "EDGES", "DEPTH"]
mode_idx = 0
screenshot = 0

# Criar pasta "img" para salvar screenshots, se não existir
IMG_DIR = "img"
os.makedirs(IMG_DIR, exist_ok=True)

# --- Pipeline ----------------------------------------------------------------
pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

xoutDisp = pipeline.create(dai.node.XLinkOut)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutDisp.setStreamName("disparity")
xoutLeft.setStreamName("left")

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setFps(30)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.setFps(30)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(True)
stereo.setSubpixel(False)
stereo.setRectifyEdgeFillColor(0)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

cfg = stereo.initialConfig.get()
cfg.costMatching.confidenceThreshold = 180
cfg.postProcessing.temporalFilter.enable = True
cfg.postProcessing.temporalFilter.persistencyMode = (
    dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4
)
cfg.postProcessing.temporalFilter.alpha = 0.7
cfg.postProcessing.temporalFilter.delta = 8
cfg.postProcessing.spatialFilter.enable = True
cfg.postProcessing.spatialFilter.holeFillingRadius = 5
cfg.postProcessing.spatialFilter.numIterations = 3
cfg.postProcessing.spatialFilter.alpha = 0.5
cfg.postProcessing.spatialFilter.delta = 10
cfg.postProcessing.thresholdFilter.minRange = 50
cfg.postProcessing.thresholdFilter.maxRange = 3000
stereo.initialConfig.set(cfg)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.disparity.link(xoutDisp.input)
stereo.rectifiedLeft.link(xoutLeft.input)

# --- Helpers -----------------------------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def process_depth(frame):
    valid_mask = frame > 0
    valid_px = frame[valid_mask]
    if len(valid_px) < 50:
        return None, None, 0.0

    lo = np.percentile(valid_px, 1)
    hi = np.percentile(valid_px, 99)
    norm = np.clip((frame - lo) / (hi - lo + 1e-6), 0, 1)
    norm[~valid_mask] = 0
    u8 = (norm * 255).astype(np.uint8)

    hole_mask = (frame == 0).astype(np.uint8)
    if hole_mask.sum() > 50:
        u8 = cv2.inpaint(u8, hole_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    u8_smooth = cv2.bilateralFilter(u8, d=9, sigmaColor=60, sigmaSpace=60)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    u8_smooth = cv2.morphologyEx(u8_smooth, cv2.MORPH_CLOSE, kernel)

    valid_pct = (valid_mask.sum() / frame.size) * 100
    return u8, u8_smooth, valid_pct

def compute_glue_mask_from_mono(mono_img):
    # mono_img: single-channel uint8
    # enhance local contrast
    enhanced = clahe.apply(mono_img)
    # high-pass to reveal subtle bands
    large_blur = cv2.GaussianBlur(enhanced, (31, 31), 0)
    high = cv2.subtract(enhanced, large_blur)
    high = cv2.normalize(high, None, 0, 255, cv2.NORM_MINMAX)

    # laplacian to emphasize edges
    lap = cv2.Laplacian(enhanced, cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)

    combined = cv2.addWeighted(high, 0.8, lap, 0.4, 0)
    _, th = cv2.threshold(combined, 18, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return closed

def overlay_mask_bgr(bgr, mask, color=(0, 0, 255), alpha=0.5):
    overlay = bgr.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)

def draw_hud(img, mode, colormap_name, dist_cm, valid_pct):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 30), (8, 8, 8), -1)
    cv2.putText(img, f"MODO: {mode}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
    cv2.putText(img, f"Valid: {valid_pct:.0f}%", (w//2-60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140,255,140), 1)
    dist_text = f"Dist: {dist_cm}cm" if dist_cm < 999 else "Dist: ---"
    cv2.putText(img, dist_text, (w-160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,100), 1)
    cv2.putText(img, "Q:sair  M:modo  C:colormap  S:screenshot", (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
    return img

# --- Main --------------------------------------------------------------------
print("OAK-D SR | Glass Glue Inspect (mono) | Q=sair M=modo C=colormap S=screenshot")

with dai.Device(pipeline) as device:
    qDisp = device.getOutputQueue("disparity", maxSize=4, blocking=False)
    qLeft = device.getOutputQueue("left", maxSize=4, blocking=False)

    while True:
        inDisp = qDisp.get()
        inLeft = qLeft.tryGet()

        raw = inDisp.getFrame().astype(np.float32)
        u8, u8_smooth, valid_pct = process_depth(raw)
        if u8 is None:
            continue

        # distancia central (approx)
        h, w = raw.shape
        cx, cy = w//2, h//2
        roi = raw[cy-30:cy+30, cx-30:cx+30]
        valid_roi = roi[roi > 0]
        dist_cm = int((460*2)/(np.median(valid_roi)+1e-6)) if len(valid_roi) > 10 else 999
        dist_cm = max(0, min(dist_cm, 999))

        # Prepare mono preview
        if inLeft is not None:
            mono = inLeft.getCvFrame()
        else:
            mono = (u8_smooth).astype(np.uint8)

        # Glue mask from mono
        # A ideia é usar o feed mono para detectar as faixas de cola, que aparecem como linhas mais escuras. O CLAHE ajuda a realçar o contraste local, e a combinação de high-pass + laplacian enfatiza essas linhas. O resultado é uma máscara binária que indica onde a cola provavelmente está presente.
        mask = compute_glue_mask_from_mono(mono if len(mono.shape)==2 else cv2.cvtColor(mono, cv2.COLOR_BGR2GRAY))
        
        # No loop principal, após mask:
        score = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]) * 100
        if score > 80:
            status = (0,255,0)  # Verde OK
        elif score > 60:
            status = (0,255,255)  # Amarelo
        else:
            status = (0,0,255)    # Vermelho FAIL

            cv2.putText(combined, f"SCORE: {score:.0f}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status, 2)
            final_img = overlay_mask_bgr(mono_bgr, mask, color=status, alpha=0.6)  # Verde=OK
            cv2.imwrite(f'final_inspect_{screenshot}.png', final_img)


        # Depth visualization
        depth_vis = cv2.applyColorMap(255 - u8_smooth, COLORMAPS[colormap_idx])

        # Enhanced mono to BGR for visualization
        mono_clahe = clahe.apply(mono) if len(mono.shape)==2 else mono
        mono_bgr = cv2.cvtColor(mono_clahe, cv2.COLOR_GRAY2BGR)

        mode = MODES[mode_idx]
        if mode == "MASK+DEPTH":
            masked = overlay_mask_bgr(mono_bgr, mask, color=(0,0,255), alpha=0.4)
            combined = np.hstack([cv2.resize(masked, (vis_width:=320, 240)), cv2.resize(depth_vis, (vis_width,240))])
        elif mode == "EDGES":
            edges = cv2.Canny(mono_clahe, 50, 150)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([cv2.resize(mono_bgr, (320,240)), cv2.resize(edges_bgr, (320,240))])
        else:
            combined = np.hstack([cv2.resize(mono_bgr, (320,240)), cv2.resize(depth_vis, (320,240))])

        combined = draw_hud(combined, mode, "", dist_cm, valid_pct)

        cv2.imshow("OAK-D SR | Inspecao Vidro", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            colormap_idx = (colormap_idx + 1) % len(COLORMAPS)
            print(f"Colormap: {colormap_idx}")
        elif key == ord('m'):
            mode_idx = (mode_idx + 1) % len(MODES)
            print(f"Modo: {MODES[mode_idx]}")
        elif key == ord('s'):
            fname = os.path.join(IMG_DIR, f"screenshot_glass_v2_{screenshot:03d}.png")
            cv2.imwrite(fname, combined)
            print(f"Screenshot salvo: {fname}")
            screenshot += 1

    cv2.destroyAllWindows()
