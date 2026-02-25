import cv2
import depthai as dai
import numpy as np

# ── Pipeline ───────────────────────────────────────────────────────────────
pipeline = dai.Pipeline()

monoLeft  = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth     = pipeline.create(dai.node.StereoDepth)
xout      = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("disparity")

# OV9282 — resolução correta para OAK-D SR
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# ── Stereo ─────────────────────────────────────────────────────────────────
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
depth.setLeftRightCheck(True)
depth.setExtendedDisparity(True)
depth.setSubpixel(False)

# Mediana forte — remove pontinhos isolados
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

config = depth.initialConfig.get()

# Filtro temporal — estabiliza entre frames
config.postProcessing.temporalFilter.enable = True
config.postProcessing.temporalFilter.persistencyMode = (
    dai.RawStereoDepthConfig.PostProcessing.TemporalFilter
    .PersistencyMode.VALID_2_IN_LAST_4
)
config.postProcessing.temporalFilter.alpha = 0.5   # mais peso no histórico = mais estável
config.postProcessing.temporalFilter.delta = 15

# Filtro espacial — preenche buracos e suaviza regiões planas
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 4   # raio maior = mais preenchimento
config.postProcessing.spatialFilter.numIterations = 2       # 2 passadas = mais limpo
config.postProcessing.spatialFilter.alpha = 0.5
config.postProcessing.spatialFilter.delta = 20

# Threshold — descarta pixels muito perto ou muito longe (evita ruído extremo)
config.postProcessing.thresholdFilter.minRange = 200    # mínimo 20 cm
config.postProcessing.thresholdFilter.maxRange = 6000   # máximo 6 m

# Decimation filter — reduz resolução pela metade e já faz hole filling interno
config.postProcessing.decimationFilter.decimationFactor = 1  # 1 = sem redução

depth.initialConfig.set(config)

monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(xout.input)

# ── Loop ───────────────────────────────────────────────────────────────────
print("OAK-D SR | Imagem limpa | 'q' para sair")

with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    max_disp = depth.initialConfig.getMaxDisparity()
    print(f"max_disp: {max_disp}")

    while True:
        inDepth = q.get()
        frame   = inDepth.getFrame().astype(np.float32)

        # ── Normalização por percentis ──────────────────────────────────────
        valid = frame[frame > 0]
        if len(valid) == 0:
            continue

        lo = np.percentile(valid, 2)
        hi = np.percentile(valid, 98)
        norm = np.clip((frame - lo) / (hi - lo + 1e-6), 0, 1)
        norm[frame == 0] = 0
        u8 = (norm * 255).astype(np.uint8)

        # ── Pós-processamento CPU ───────────────────────────────────────────

        # 1. Inpainting — preenche buracos usando pixels vizinhos
        mask = (u8 == 0).astype(np.uint8)
        inpainted = cv2.inpaint(u8, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        # 2. Bilateral — suaviza ruído preservando bordas de profundidade
        smooth = cv2.bilateralFilter(inpainted, d=9, sigmaColor=60, sigmaSpace=60)

        # 3. Colormap JET
        jet = cv2.applyColorMap(smooth, cv2.COLORMAP_JET)

        cv2.imshow("Depth", jet)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()