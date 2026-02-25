import cv2
import depthai as dai
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  OAK-D SR — INSPEÇÃO DE DEFEITOS (roda de carro)
#  Modos: D = depth normal | E = edge/defeitos realçados | B = blend
#  Teclas: Q = sair | C = colormap | M = modo | S = screenshot
# ══════════════════════════════════════════════════════════════════════════════

COLORMAPS      = [cv2.COLORMAP_TURBO, cv2.COLORMAP_JET, cv2.COLORMAP_MAGMA]
COLORMAP_NAMES = ["TURBO", "JET", "MAGMA"]
colormap_idx   = 0

MODES      = ["DEPTH", "DEFEITOS", "BLEND"]
mode_idx   = 0
screenshot = 0

# ── Pipeline ──────────────────────────────────────────────────────────────────
pipeline = dai.Pipeline()

monoLeft  = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo    = pipeline.create(dai.node.StereoDepth)
xoutDisp  = pipeline.create(dai.node.XLinkOut)
xoutLeft  = pipeline.create(dai.node.XLinkOut)
xoutDisp.setStreamName("disparity")
xoutLeft.setStreamName("left")

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setFps(30)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.setFps(30)

# ── Stereo — otimizado para objetos grandes e próximos ────────────────────────
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(True)
stereo.setSubpixel(False)
stereo.setRectifyEdgeFillColor(0)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

config = stereo.initialConfig.get()
# Threshold mais baixo = mais pixels válidos na superfície da roda
config.costMatching.confidenceThreshold = 180

# Temporal mais agressivo = imagem mais estável para inspeção
config.postProcessing.temporalFilter.enable = True
config.postProcessing.temporalFilter.persistencyMode = (
    dai.RawStereoDepthConfig.PostProcessing.TemporalFilter
    .PersistencyMode.VALID_2_IN_LAST_4
)
config.postProcessing.temporalFilter.alpha = 0.7  # mais peso no histórico
config.postProcessing.temporalFilter.delta = 8

# Espacial forte = superfície da roda mais contínua
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 5
config.postProcessing.spatialFilter.numIterations = 3
config.postProcessing.spatialFilter.alpha = 0.5
config.postProcessing.spatialFilter.delta = 10  # delta menor = preserva mais detalhes

config.postProcessing.thresholdFilter.minRange = 100
config.postProcessing.thresholdFilter.maxRange = 3000

stereo.initialConfig.set(config)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.disparity.link(xoutDisp.input)
stereo.rectifiedLeft.link(xoutLeft.input)

# ── Objetos reutilizáveis (fora do loop!) ─────────────────────────────────────
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def process_depth(frame):
    """Normaliza e suaviza o frame de disparidade."""
    valid_mask = frame > 0
    valid_px   = frame[valid_mask]
    valid_pct  = (valid_mask.sum() / frame.size) * 100

    if len(valid_px) < 100:
        return None, None, valid_pct

    # Normalização por percentis — estica contraste para o que está na cena
    lo   = np.percentile(valid_px, 1)
    hi   = np.percentile(valid_px, 99)
    norm = np.clip((frame - lo) / (hi - lo + 1e-6), 0, 1)
    norm[~valid_mask] = 0
    u8   = (norm * 255).astype(np.uint8)

    # Inpainting só se tiver buracos relevantes (evita custo desnecessário)
    hole_mask = (frame == 0).astype(np.uint8)
    if hole_mask.sum() > 50:
        u8 = cv2.inpaint(u8, hole_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # Bilateral — suaviza ruído preservando bordas (bom para detectar defeitos)
    u8_smooth = cv2.bilateralFilter(u8, d=11, sigmaColor=60, sigmaSpace=60)

    # Closing — fecha pequenos buracos restantes
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    u8_smooth = cv2.morphologyEx(u8_smooth, cv2.MORPH_CLOSE, kernel)

    return u8, u8_smooth, valid_pct


def detect_defects(u8_smooth):
    """
    Realça variações locais de profundidade — defeitos aparecem como
    regiões brilhantes sobre fundo escuro.
    """
    # Gradiente local (Sobel) — detecta mudanças abruptas de profundidade
    grad_x = cv2.Sobel(u8_smooth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(u8_smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad   = cv2.magnitude(grad_x, grad_y)
    grad   = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # CLAHE no gradiente — realça defeitos pequenos E grandes
    grad_clahe = clahe.apply(grad)

    # Limiar adaptativo — isola regiões com variação acima do esperado
    thresh = cv2.adaptiveThreshold(
        grad_clahe, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, -5
    )

    # Fecha pequenas falhas nos contornos dos defeitos
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh   = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_d)

    # Colormap quente para defeitos — vermelho = defeito, preto = ok
    defect_color = cv2.applyColorMap(thresh, cv2.COLORMAP_HOT)

    return defect_color, thresh


def draw_hud(img, mode, colormap_name, dist_cm, valid_pct):
    h, w = img.shape[:2]

    # Barra topo
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, 38), (10, 10, 10), -1)
    cv2.addWeighted(bar, 0.55, img, 0.45, 0, img)

    cv2.putText(img, f"MODO: {mode}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Validos: {valid_pct:.0f}%", (w // 2 - 60, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 255, 140), 1, cv2.LINE_AA)
    dist_text  = f"Dist: {dist_cm}cm" if dist_cm < 999 else "Dist: ---"
    dist_color = (100, 255, 100) if dist_cm < 999 else (80, 80, 255)
    cv2.putText(img, dist_text, (w - 155, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 1, cv2.LINE_AA)

    # Mira central
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx - 15, cy), (cx + 15, cy), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 15), (cx, cy + 15), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 24, (255, 255, 255), 1, cv2.LINE_AA)

    # Rodapé
    bar2 = img.copy()
    cv2.rectangle(bar2, (0, h - 28), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(bar2, 0.55, img, 0.45, 0, img)
    cv2.putText(img, "Q:sair  M:modo  C:colormap  S:screenshot",
                (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
    return img


# ── Main loop ─────────────────────────────────────────────────────────────────
print("OAK-D SR | Inspeção de defeitos | Q=sair M=modo C=colormap S=screenshot")

with dai.Device(pipeline) as device:
    qDisp = device.getOutputQueue("disparity", maxSize=4, blocking=False)
    qLeft = device.getOutputQueue("left",      maxSize=4, blocking=False)

    max_disp = stereo.initialConfig.getMaxDisparity()
    print(f"max_disp: {max_disp:.0f}")

    while True:
        inDisp = qDisp.get()
        inLeft = qLeft.tryGet()

        raw = inDisp.getFrame().astype(np.float32)
        u8, u8_smooth, valid_pct = process_depth(raw)
        if u8 is None:
            continue

        # Distância central
        h, w = raw.shape
        cx, cy = w // 2, h // 2
        roi       = raw[cy-30:cy+30, cx-30:cx+30]
        valid_roi = roi[roi > 0]
        dist_cm   = int((460 * 2) / (np.median(valid_roi) + 1e-6)) if len(valid_roi) > 10 else 999
        dist_cm   = max(0, min(dist_cm, 999))

        # ── Modo de visualização ──────────────────────────────────────────────
        mode = MODES[mode_idx]

        if mode == "DEPTH":
            # Depth normal com colormap invertido (perto=vermelho, longe=azul)
            vis = cv2.applyColorMap(255 - u8_smooth, COLORMAPS[colormap_idx])

        elif mode == "DEFEITOS":
            # Gradiente de profundidade — realça bordas e irregularidades
            defect_color, _ = detect_defects(u8_smooth)
            vis = defect_color

        else:  # BLEND
            # Depth + defeitos sobrepostos — melhor para inspecionar
            depth_vis       = cv2.applyColorMap(255 - u8_smooth, COLORMAPS[colormap_idx])
            defect_color, thresh = detect_defects(u8_smooth)
            # Só mostra defeitos onde o gradiente é forte
            defect_mask = (thresh > 0).astype(np.uint8)
            defect_mask_3ch = cv2.merge([defect_mask, defect_mask, defect_mask])
            vis = np.where(defect_mask_3ch == 1,
                           cv2.addWeighted(depth_vis, 0.3, defect_color, 0.7, 0),
                           depth_vis)

        vis = draw_hud(vis, mode, COLORMAP_NAMES[colormap_idx], dist_cm, valid_pct)

        # ── Layout: mono + depth ──────────────────────────────────────────────
        if inLeft is not None:
            left_img = inLeft.getCvFrame()
            left_img = clahe.apply(left_img)
            left_rgb = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
            left_rgb = cv2.resize(left_rgb, (vis.shape[1], vis.shape[0]))

            cv2.putText(left_rgb, "MONO", (8, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
            cv2.putText(vis, mode, (8, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)

            divider  = np.full((vis.shape[0], 2, 3), 60, dtype=np.uint8)
            combined = np.hstack([left_rgb, divider, vis])
        else:
            combined = vis

        cv2.imshow("OAK-D SR | Inspeção", combined)

        # ── Teclas ────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            colormap_idx = (colormap_idx + 1) % len(COLORMAPS)
            print(f"Colormap: {COLORMAP_NAMES[colormap_idx]}")
        elif key == ord('m'):
            mode_idx = (mode_idx + 1) % len(MODES)
            print(f"Modo: {MODES[mode_idx]}")
        elif key == ord('s'):
            fname = f"screenshot_{screenshot:03d}.png"
            cv2.imwrite(fname, combined)
            print(f"Screenshot salvo: {fname}")
            screenshot += 1

cv2.destroyAllWindows()
print("Encerrado.")
