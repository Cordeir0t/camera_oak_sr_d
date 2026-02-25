import cv2
import depthai as dai
import numpy as np
import time
from collections import deque

WINDOW_NAME = "OAK-D Cola no Vidro - Medidor v4.2 (Espessura Corrigida)"
QUIT_KEY = ord('q')
TOGGLE_KEY = ord('t')
RESET_KEY = ord('r')
INFO_KEY = ord('i')
ROI_KEY = ord('s')

FORCE_LOW_BANDWIDTH = True

class ColaMeasurer:
    def __init__(self):
        self.pipeline = None
        self.disparity_queue = None
        self.depth_queue = None
        
        self.depth_history = deque(maxlen=8)
        self.alpha_temporal = 0.12
        
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = deque(maxlen=30)
        self.max_depth_mm = 1000
        
        self.show_depth = True
        self.show_hud = True
        
        self.roi = None
        self.altura_mm = 0.0  # ESPESSURA da cola
        self.largura_mm = 0.0
        self.focal_px = 882   # será ajustado com calibração
        
        # Visual
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.gamma = 0.7
        cmap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
        cmap[0] = [0, 0, 0]
        self.color_map = cmap
        
        self.setup_pipeline()

    def setup_pipeline(self):
        """Pipeline DepthAI 3.2.1 - API moderna + configs otimizadas"""
        pipeline = dai.Pipeline()
        
        # Câmeras mono full-res NV12
        mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        
        mono_left_out = mono_left.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
        mono_right_out = mono_right.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
        
        stereo = pipeline.create(dai.node.StereoDepth)
        mono_left_out.link(stereo.left)
        mono_right_out.link(stereo.right)
        
        # Nomear streams para queues
        stereo.disparity.out.setStreamName("disp")
        stereo.depth.out.setStreamName("depth")
        
        # Preset otimizado
        if FORCE_LOW_BANDWIDTH:
            preset = dai.node.StereoDepth.PresetMode.FAST_DENSITY
        else:
            preset = dai.node.StereoDepth.PresetMode.HIGH_DENSITY
        stereo.setDefaultProfilePreset(preset)
        
        # Configs finas
        cfg = stereo.initialConfig
        cfg.postProcessing.spatialFilter.enable = True
        cfg.postProcessing.spatialFilter.holeFillingRadius = 1
        
        try:
            cfg.postProcessing.temporalFilter.enable = False
            cfg.postProcessing.decimationFilter.enable = True
            cfg.postProcessing.decimationFilter.decimationFactor = 4
        except Exception:
            pass
            
        cfg.postProcessing.thresholdFilter.minRange = 100
        cfg.postProcessing.thresholdFilter.maxRange = 1500
        
        # Confiança mais alta = menos ruído
        try:
            cfg.setConfidenceThreshold(230)
        except Exception:
            pass
            
        stereo.initialConfig = cfg
        
        self.pipeline = pipeline

    def mouse_callback(self, event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.roi is None:
            roi_w, roi_h = 120, 120
            x = max(0, mx - roi_w//2)
            y = max(0, my - roi_h//2)
            x = min(x, 800 - roi_w)
            y = min(y, 600 - roi_h)
            self.roi = (x, y, roi_w, roi_h)
            print(f"ROI definida: {self.roi}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.roi = None
            self.altura_mm = 0
            self.largura_mm = 0
            print("ROI resetada")

    def medir_cola(self, depth_frame):
        """Medição corrigida: espessura (altura) da cola em mm"""
        if self.roi is None:
            return

        h_img, w_img = depth_frame.shape
        x, y, w, h = self.roi

        x = max(0, min(x, w_img - w))
        y = max(0, min(y, h_img - h))
        self.roi = (x, y, w, h)

        roi_depth = depth_frame[y:y+h, x:x+w].astype(np.float32)
        valid_mask = roi_depth > 50

        if np.sum(valid_mask) < 100:
            self.altura_mm = 0.0
            self.largura_mm = 0.0
            return

        # Plano do vidro (mediana dos pixels válidos na ROI)
        vidro_medio = np.median(roi_depth[valid_mask])
        diff_full = vidro_medio - roi_depth
        cola_mask_full = (diff_full > self.min_altura_threshold) & valid_mask

        if np.sum(cola_mask_full) < 30:
            self.altura_mm = 0.0
            self.largura_mm = 0.0
            return

        # ESPESSURA = mediana do ressalto da cola
        self.altura_mm = float(np.median(diff_full[cola_mask_full]))

        # Largura usando contorno
        roi_full_mask = np.zeros((h, w), dtype=np.uint8)
        rows, cols = np.where(cola_mask_full)
        sel_rows = rows[:100]
        sel_cols = cols[:100]
        roi_full_mask[sel_rows, sel_cols] = 255

        contours, _ = cv2.findContours(roi_full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            bbox = cv2.boundingRect(largest)
            z_medio = float(vidro_medio - np.median(diff_full[cola_mask_full]))
            self.largura_mm = bbox[2] * z_medio / self.focal_px
        else:
            self.largura_mm = 0.0

    def apply_temporal_filter(self, frame):
        self.depth_history.append(frame)
        if len(self.depth_history) >= 3:
            stack = np.stack(list(self.depth_history), axis=0).astype(np.float32)
            med = np.median(stack, axis=0)
            return np.clip(med, 0, 255).astype(np.uint8)
        elif len(self.depth_history) > 1:
            prev = np.array(self.depth_history[-2], dtype=np.float32)
            current = np.array(self.depth_history[-1], dtype=np.float32)
            sm = self.alpha_temporal * current + (1 - self.alpha_temporal) * prev
            return np.clip(sm, 0, 255).astype(np.uint8)
        return frame

    def enhance_visual(self, normalized):
        enhanced = self.clahe.apply(normalized)
        enhanced = np.power(enhanced / 255.0, self.gamma)
        enhanced = (enhanced * 255).astype(np.uint8)
        enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        enhanced = cv2.addWeighted(enhanced, 1.2, blurred, -0.2, 0)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.05, beta=5)
        return enhanced

    def get_fps(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            fps = self.frame_count / elapsed
            self.fps_history.append(fps)
            self.frame_count = 0
            self.start_time = time.time()
            return np.mean(self.fps_history)
        return 0.0

    def update_max_range(self, frame):
        valid_pixels = frame[frame > 0]
        if len(valid_pixels) > 0:
            self.max_depth_mm = np.percentile(valid_pixels, 95)

    def draw_hud(self, frame):
        fps = self.get_fps()
        h_img, w_img = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Modo: {'DEPTH' if self.show_depth else 'DISP'}", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Max: {self.max_depth_mm:.0f}mm", (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if self.altura_mm > 0:
            cor = (0, 255, 0)
            status = "OK"
        else:
            cor = (0, 165, 255)
            status = "--"
        cv2.putText(frame, f"Cola LxE: {self.largura_mm:.1f}x{self.altura_mm:.1f}mm [{status}]", 
                   (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

        if self.roi:
            rx, ry, rw, rh = self.roi
            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 0, 255), 2)

        cv2.putText(frame, "[S]ROI [T]Modo [R]Reset [I]HUD [Q]Sair  Clique esquerdo=ROI", 
                   (10, h_img-25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    def process_frame(self, disp_frame, depth_frame):
        depth_raw = depth_frame.astype(np.uint16)
        self.update_max_range(depth_raw)
        self.medir_cola(depth_raw)

        if self.show_depth:
            normalized = np.clip((depth_raw / max(1, self.max_depth_mm) * 255), 0, 255).astype(np.uint8)
        else:
            normalized = np.clip((disp_frame * 1.35), 0, 255).astype(np.uint8)

        filtered = self.apply_temporal_filter(normalized)
        if filtered.dtype != np.uint8:
            filtered = filtered.astype(np.uint8)
        if len(filtered.shape) != 2:
            filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

        enhanced = self.enhance_visual(filtered)
        colorized = cv2.applyColorMap(enhanced, self.color_map)

        if self.show_hud:
            self.draw_hud(colorized)
        return colorized

    def run(self):
        with dai.Device(self.pipeline) as device:
            disp_q = device.getOutputQueue("disp", maxSize=1, blocking=False)
            depth_q = device.getOutputQueue("depth", maxSize=1, blocking=False)

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1280, 720)
            cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

            frame_skip = 0
            skip_rate = 5 if FORCE_LOW_BANDWIDTH else 2

            while True:
                frame_skip += 1
                if frame_skip % skip_rate != 0:
                    if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                        break
                    continue

                disp = disp_q.tryGet()
                depth = depth_q.tryGet()

                if disp is None or depth is None:
                    continue

                colorized = self.process_frame(disp.getFrame(), depth.getFrame())
                cv2.imshow(WINDOW_NAME, colorized)

                key = cv2.waitKey(1) & 0xFF
                if key == QUIT_KEY:
                    break
                elif key == TOGGLE_KEY:
                    self.show_depth = not self.show_depth
                elif key == RESET_KEY:
                    self.depth_history.clear()
                    self.max_depth_mm = 1000
                    self.roi = None
                    self.altura_mm = 0
                    self.largura_mm = 0
                elif key == INFO_KEY:
                    self.show_hud = not self.show_hud
                elif key == ROI_KEY:
                    self.roi = None

        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("OAK-D Medidor de Cola v4.2 - ESPESSURA corrigida")
    print("Clique esquerdo na cola para medir LxE em mm")
    print("Controles: [R]Reset [Q]Sair [T]Depth/Disparity")
    
    app = ColaMeasurer()
    app.run()
