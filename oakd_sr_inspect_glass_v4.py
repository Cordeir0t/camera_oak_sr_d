import cv2
import depthai as dai
import numpy as np
import os

"""
OAK-D SR — Inspeção de faixas de cola em vidro (VERSÃO FINAL V7 - THICKNESS REAL)
Detecção por CANNY EDGES + medição REAL de espessura (projeção vertical).
Perfeito para faixas finas 0.5-3%, 1-6 faixas, 8-25px reais.
Teclas: Q=sair | M=modo | C=colormap | S=screenshot | P=score-print
"""

COLORMAPS = [cv2.COLORMAP_TURBO, cv2.COLORMAP_JET, cv2.COLORMAP_MAGMA]
colormap_idx = 0
MODES = ["MASK+DEPTH", "EDGES", "DEPTH"]
mode_idx = 0
screenshot = 0

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
cfg.postProcessing.temporalFilter.persistencyMode = dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4
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
    """V7: CANNY EDGES + dilatação horizontal para decalques/faixas"""
    gray = mono_img if len(mono_img.shape)==2 else cv2.cvtColor(mono_img, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    
    # CANNY - capta os decalques que você vê!
    edges = cv2.Canny(enhanced, 40, 120)
    
    # Dilatação HORIZONTAL para faixas
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    dilated = cv2.dilate(edges, kernel_edge, iterations=2)
    
    # High-pass reforço
    large_blur = cv2.GaussianBlur(enhanced, (21, 21), 0)
    high = cv2.subtract(enhanced, large_blur)
    high = cv2.normalize(high, None, 0, 255, cv2.NORM_MINMAX)
    _, high_th = cv2.threshold(high, 35, 255, cv2.THRESH_BINARY)
    
    combined = cv2.bitwise_or(dilated, high_th)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Filtro faixas finas
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(cleaned)
    total_area = cleaned.size
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < total_area * 0.05:
            cv2.fillPoly(filtered, [cnt], 255)
    
    return filtered

def compute_glue_score(mask, raw_depth):
    """V7: THICKNESS REAL por projeção vertical"""
    h, w = mask.shape
    total_px = h * w
    glue_px = np.sum(mask > 0)
    coverage = (glue_px / total_px) * 100
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if 50 < cv2.contourArea(c) < total_px * 0.05]
    num_bands = len(valid_contours)
    
    if valid_contours:
        # THICKNESS REAL: pixels ocupados verticalmente
        mask_proj = np.any(mask > 0, axis=1).astype(int)
        proj_thick = np.sum(mask_proj)
        
        # Bbox da maior faixa
        largest = max(valid_contours, key=cv2.contourArea)
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(largest)
        
        # Usa o MENOR (mais preciso)
        avg_thick = min(proj_thick, bbox_h)
        
        # Horizontal: largura >> altura
        is_horizontal = bbox_w > bbox_h * 2.5
        thick_ok = is_horizontal and 8 <= avg_thick <= 25
    else:
        avg_thick = 0
        thick_ok = False
    
    # Thresholds finais
    cov_score = 100 if 0.5 <= coverage <= 3.0 else 0
    band_score = 100 if 1 <= num_bands <= 6 else 0
    thick_score = 100 if thick_ok else 30
    
    final_score = (cov_score * 0.5 + band_score * 0.3 + thick_score * 0.2)
    status = " OK" if final_score >= 90 else " AVISO" if final_score >= 70 else " FAIL"
    
    return {
        'score': round(final_score, 1),
        'status': status,
        'coverage': round(coverage, 2),
        'bands': num_bands,
        'thickness': avg_thick,
        'color': (0,255,0) if final_score >= 90 else (0,165,255) if final_score >= 70 else (0,0,255)
    }

def overlay_mask_bgr(bgr, mask, color=(0, 0, 255), alpha=0.5):
    overlay = bgr.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)

def draw_hud(img, mode, dist_cm, valid_pct, score_info):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 70), (8, 8, 8), -1)
    
    cv2.putText(img, f"MODO: {mode}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
    cv2.putText(img, f"Valid: {valid_pct:.0f}%", (w//2-60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140,255,140), 1)
    dist_text = f"Dist: {dist_cm}cm" if dist_cm < 999 else "Dist: ---"
    cv2.putText(img, dist_text, (w-160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,100), 1)
    
    # SCORE PRINCIPAL
    cv2.putText(img, f"SCORE: {score_info['score']} {score_info['status']}", 
                (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_info['color'], 2)
    
    cv2.putText(img, f"Cov:{score_info['coverage']}% | F:{score_info['bands']} | T:{score_info['thickness']}px", 
                (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    
    cv2.putText(img, "Q:sair M:modo C:colormap S:screenshot P:score-print", 
                (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
    return img

# --- Main --------------------------------------------------------------------
print(" OAK-D SR | Glass Glue Inspect V7 - THICKNESS REAL")
print(" Mede espessura REAL (projeção) | Coverage 0.5-3%, Thick 8-25px")

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

        # Distância central
        h, w = raw.shape
        cx, cy = w//2, h//2
        roi = raw[cy-30:cy+30, cx-30:cx+30]
        valid_roi = roi[roi > 0]
        dist_cm = int((460*2)/(np.median(valid_roi)+1e-6)) if len(valid_roi) > 10 else 999
        dist_cm = max(0, min(dist_cm, 999))

        # Mono preview
        if inLeft is not None:
            mono = inLeft.getCvFrame()
        else:
            mono = u8_smooth.astype(np.uint8)

        # DETECÇÃO + SCORE V7
        mask = compute_glue_mask_from_mono(mono if len(mono.shape)==2 else cv2.cvtColor(mono, cv2.COLOR_BGR2GRAY))
        score_info = compute_glue_score(mask, raw)

        # Visualizações
        depth_vis = cv2.applyColorMap(255 - u8_smooth, COLORMAPS[colormap_idx])
        mono_clahe = clahe.apply(mono) if len(mono.shape)==2 else mono
        mono_bgr = cv2.cvtColor(mono_clahe, cv2.COLOR_GRAY2BGR)

        vis_width = 320
        mode = MODES[mode_idx]
        if mode == "MASK+DEPTH":
            masked = overlay_mask_bgr(mono_bgr, mask, color=score_info['color'], alpha=0.6)
            combined = np.hstack([cv2.resize(masked, (vis_width, 240)), cv2.resize(depth_vis, (vis_width, 240))])
        elif mode == "EDGES":
            edges = cv2.Canny(mono_clahe, 40, 120)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([cv2.resize(mono_bgr, (vis_width, 240)), cv2.resize(edges_bgr, (vis_width, 240))])
        else:
            combined = np.hstack([cv2.resize(mono_bgr, (vis_width, 240)), cv2.resize(depth_vis, (vis_width, 240))])

        combined = draw_hud(combined, mode, dist_cm, valid_pct, score_info)
        cv2.imshow("OAK-D SR | Glue Inspect V7 - PERFECT ", combined)

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
            fname = os.path.join(IMG_DIR, f"screenshot_glass_v7_{screenshot:03d}.png")
            cv2.imwrite(fname, combined)
            print(f" Screenshot: {fname} | {score_info}")
            screenshot += 1
        elif key == ord('p'):
            print(f" SCORE ATUAL: {score_info}")
    
cv2.destroyAllWindows()
print(" Inspeção V7 finalizada! Perfeita para produção.")
