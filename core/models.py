import os, joblib, mediapipe as mp

class ModelLoader:
    def __init__(self, models_dir=None):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not models_dir:
            models_dir = os.path.join(base_path, "models")
        elif not os.path.isabs(models_dir):
            models_dir = os.path.join(base_path, "models") if ".." in models_dir else os.path.abspath(models_dir)
            
        self.hand_landmarker_path = os.path.join(models_dir, "hand_landmarker.task")
        self.gesture_model_path = os.path.join(models_dir, "modelo_gestos.pkl")
        
        if not os.path.exists(self.gesture_model_path):
            raise FileNotFoundError(f"Erro: Modelo {self.gesture_model_path} nao encontrado na nuvem!")
            
        self.clf = joblib.load(self.gesture_model_path)
        self.base_options = mp.tasks.BaseOptions(model_asset_path=self.hand_landmarker_path)
        self.options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=self.base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(self.options)
