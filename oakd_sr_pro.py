import cv2
import depthai as dai
import numpy as np



COLORMAPS      = [cv2.COLORMAP_TURBO, cv2.COLORMAP_JET, cv2.COLORMAP_MAGMA, cv2.COLORMAP_VIRIDIS]
COLORMAP_NAMES = ["TURBO", "JET", "MAGMA", "VIRIDIS"]
colormap_idx   = 0

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

# ── Stereo ────────────────────────────────────────────────────────────────────
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(True)
stereo.setSubpixel(False)
stereo.setRectifyEdgeFillColor(0)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

config = stereo.initialConfig.get()
config.costMatching.confidenceThreshold = 200

# Temporal: mais estável, menos flickering
config.postProcessing.temporalFilter.enable = True
config.postProcessing.temporalFilter.persistencyMode = (
    dai.RawStereoDepthConfig.PostProcessing.TemporalFilter
    .PersistencyMode.VALID_2_IN_LAST_4
)
config.postProcessing.temporalFilter.alpha = 0.6  # mais suave
config.postProcessing.temporalFilter.delta = 10

# Espacial: hole filling mais agressivo
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 4
config.postProcessing.spatialFilter.numIterations = 3  # 3 passadas
config.postProcessing.spatialFilter.alpha = 0.5
config.postProcessing.spatialFilter.delta = 15

config.postProcessing.thresholdFilter.minRange = 100
config.postProcessing.thresholdFilter.maxRange = 3000

stereo.initialConfig.set(config)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.disparity.link(xoutDisp.input)
stereo.rectifiedLeft.link(xoutLeft.input)

# ── Pós-processamento CPU ─────────────────────────────────────────────────────
def process_depth(frame):
    valid_mask = frame > 0
    valid_px   = frame[valid_mask]
    valid_pct  = (valid_mask.sum() / frame.size) * 100

    if len(valid_px) < 100:
        return None, valid_pct

    # Normalização por percentis
    lo   = np.percentile(valid_px, 1)
    hi   = np.percentile(valid_px, 99)
    norm = np.clip((frame - lo) / (hi - lo + 1e-6), 0, 1)
    norm[~valid_mask] = 0
    u8   = (norm * 255).astype(np.uint8)

    # 1. Inpainting — preenche buracos usando contexto dos vizinhos
    hole_mask = (frame == 0).astype(np.uint8)
    if hole_mask.sum() > 0:
        u8 = cv2.inpaint(u8, hole_mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)

    # 2. Bilateral — suaviza sem borrar bordas de profundidade
    u8 = cv2.bilateralFilter(u8, d=9, sigmaColor=50, sigmaSpace=50)

    # 3. Closing morfológico — fecha pequenos buracos que sobraram
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)

    # 4. Leve gaussian para suavizar bordas irregulares
    u8 = cv2.GaussianBlur(u8, (3, 3), 0)

    return u8, valid_pct

def draw_hud(img, colormap_name, dist_cm, valid_pct):
    h, w = img.shape[:2]

    # Barra topo semi-transparente
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, 36), (10, 10, 10), -1)
    cv2.addWeighted(bar, 0.6, img, 0.4, 0, img)

    cv2.putText(img, f"Colormap: {colormap_name}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Validos: {valid_pct:.0f}%", (w // 2 - 55, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 255, 140), 1, cv2.LINE_AA)

    dist_color = (100, 255, 100) if dist_cm < 999 else (100, 100, 255)
    dist_text  = f"Dist: {dist_cm}cm" if dist_cm < 999 else "Dist: ---"
    cv2.putText(img, dist_text, (w - 150, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 1, cv2.LINE_AA)

    # Mira central
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx - 14, cy), (cx + 14, cy), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 14), (cx, cy + 14), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 22, (255, 255, 255), 1, cv2.LINE_AA)

    # Barra rodapé
    bar2 = img.copy()
    cv2.rectangle(bar2, (0, h - 26), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(bar2, 0.6, img, 0.4, 0, img)
    cv2.putText(img, "Q: sair  |  C: colormap  |  OAK-D SR",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
    return img

# ── Main loop ─────────────────────────────────────────────────────────────────
print("OAK-D SR | 30 FPS | Q = sair | C = colormap")

with dai.Device(pipeline) as device:
    qDisp = device.getOutputQueue("disparity", maxSize=4, blocking=False)
    qLeft = device.getOutputQueue("left",      maxSize=4, blocking=False)

    max_disp = stereo.initialConfig.getMaxDisparity()
    print(f"max_disp: {max_disp:.0f}")

    while True:
        inDisp = qDisp.get()
        inLeft = qLeft.tryGet()

        raw = inDisp.getFrame().astype(np.float32)
        u8, valid_pct = process_depth(raw)
        if u8 is None:
            continue

        # Distância central
        h, w = raw.shape
        cx, cy = w // 2, h // 2
        roi = raw[cy-30:cy+30, cx-30:cx+30]
        valid_roi = roi[roi > 0]
        if len(valid_roi) > 10:
            dist_cm = int((460 * 2) / (np.median(valid_roi) + 1e-6))
            dist_cm = max(0, min(dist_cm, 999))
        else:
            dist_cm = 999

        # Colormap + HUD
        depth_colored = cv2.applyColorMap(u8, COLORMAPS[colormap_idx])
        depth_colored = draw_hud(depth_colored, COLORMAP_NAMES[colormap_idx],
                                 dist_cm, valid_pct)

        # Layout lado a lado
        if inLeft is not None:
            left_img = inLeft.getCvFrame()
            # CLAHE na mono — mais contraste e detalhe
            clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            left_img = clahe.apply(left_img)
            left_rgb = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
            left_rgb = cv2.resize(left_rgb, (depth_colored.shape[1], depth_colored.shape[0]))

            # Labels
            cv2.putText(left_rgb, "MONO", (8, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
            cv2.putText(depth_colored, "DEPTH", (8, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)

            divider  = np.full((depth_colored.shape[0], 2, 3), 50, dtype=np.uint8)
            combined = np.hstack([left_rgb, divider, depth_colored])
            cv2.imshow("OAK-D SR", combined)
        else:
            cv2.imshow("OAK-D SR", depth_colored)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            colormap_idx = (colormap_idx + 1) % len(COLORMAPS)
            print(f"Colormap: {COLORMAP_NAMES[colormap_idx]}")

cv2.destroyAllWindows()
print("Encerrado.")