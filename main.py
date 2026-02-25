# OAK-D Stereo Depth - JLR Foam Thickness Monitor v2.0
# - Pipeline otimizada para estabilidade e desempenho
# - Filtro temporal para suavizar o depth
# - HUD informativo com FPS, modo e estatísticas
# - Teclas: [T] Toggle Depth/Disparity, [R] Reset, [I] Toggle HUD, [Q] Quit
# depthau 3.2.1, Python 3.10 (exencial), OpenCV 4.5+ recomendado para funcionamento ideal de acordo com as API's que funcionam melhro com a camera OAK-D SR 


import cv2
import depthai as dai
import numpy as np
import time
from collections import deque

WINDOW_NAME = "OAK-D Stereo Depth - JLR Foam Thickness Monitor v2.0"
QUIT_KEY = ord('q')
TOGGLE_KEY = ord('t')
RESET_KEY = ord('r')
INFO_KEY = ord('i')

class StereoDepthApp:
    def __init__(self):
        self.pipeline = None
        self.disparity_queue = None
        self.depth_queue = None

        self.depth_history = deque(maxlen=5)
        self.alpha_temporal = 0.3

        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = deque(maxlen=30)
        self.max_depth_mm = 1000

        self.show_depth = True
        self.show_hud = True
        self.color_map = self._create_colormap()
        
        self.setup_pipeline()
        
    def _create_colormap(self):
        color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
        color_map[0] = [0, 0, 0]
        return color_map
    
    def setup_pipeline(self):
        pipeline = dai.Pipeline()
        
        #  SEM setFpsLimit (não existe em Camera v3)
        mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        
        mono_left_out = mono_left.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
        mono_right_out = mono_right.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
        
        stereo = pipeline.create(dai.node.StereoDepth)
        mono_left_out.link(stereo.left)
        mono_right_out.link(stereo.right)
        
        # Config otimizada anti-freeze
        stereo.initialConfig.setExtendedDisparity(True)
        stereo.initialConfig.setLeftRightCheck(False)
        stereo.initialConfig.setSubpixel(False)
        stereo.initialConfig.setConfidenceThreshold(250)
        
        # Queues anti-travamento
        self.disparity_queue = stereo.disparity.createOutputQueue(maxSize=2, blocking=False)
        self.depth_queue = stereo.depth.createOutputQueue(maxSize=2, blocking=False)
        
        self.pipeline = pipeline
    
    def apply_temporal_filter(self, frame):
        if self.depth_history:
            prev = np.array(self.depth_history[-1])
            frame = self.alpha_temporal * frame + (1 - self.alpha_temporal) * prev
        self.depth_history.append(frame)
        return frame
    
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
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Mode: {'DEPTH (mm)' if self.show_depth else 'DISPARITY'}",
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Max range: {self.max_depth_mm:.0f}mm", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Valid pixels: {np.sum(frame > 0):,}",
                   (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, "[T] Toggle  [R] Reset  [I] HUD  [Q] Quit",
                   (10, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def process_frame(self, disp_frame, depth_frame):
        if self.show_depth:
            frame = depth_frame.astype(np.uint16)
            self.update_max_range(frame)
            normalized = np.clip((frame / self.max_depth_mm * 255).astype(np.uint8), 0, 255)
        else:
            frame = disp_frame
            normalized = np.clip((frame * 1.35).astype(np.uint8), 0, 255)

        filtered = self.apply_temporal_filter(normalized)
        if filtered.dtype != np.uint8 or len(filtered.shape) != 2:
            filtered = filtered.astype(np.uint8)

        colorized = cv2.applyColorMap(filtered, self.color_map)
        if self.show_hud:
            self.draw_hud(colorized)
        return colorized

    def run(self):
        with self.pipeline:
            self.pipeline.start()

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1280, 720)

            frame_skip = 0

            while self.pipeline.isRunning():
                frame_skip += 1
                if frame_skip % 2 != 0:  # 30 FPS suave
                    continue

                disp = self.disparity_queue.tryGet()
                depth = self.depth_queue.tryGet()

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
                elif key == INFO_KEY:
                    self.show_hud = not self.show_hud

            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = StereoDepthApp()
    app.run()
