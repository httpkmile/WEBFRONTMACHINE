import cv2, time, mediapipe as mp
from .models import ModelLoader
from .processor import FrameProcessor

class WebcamRecognizer:
    def __init__(self, models_dir="models"):
        self.loader = ModelLoader(models_dir=models_dir)
        self.processor = FrameProcessor()

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            ts = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self.loader.landmarker.detect_for_video(mp_image, ts)
            cv2.imshow('Gesture Recognition', self.processor.process_frame(frame, result, self.loader.clf))
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
