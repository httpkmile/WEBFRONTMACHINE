import os
import joblib
import mediapipe as mp

class ModelLoader:
    """Carrega e configura os modelos de IA."""
    def __init__(self, models_dir="../models"):
        # Caminhos dos modelos
        self.hand_landmarker_path = os.path.join(models_dir, "hand_landmarker.task")
        self.gesture_model_path = os.path.join(models_dir, "modelo_gestos.pkl")
        
        # Ajuste para quando chamado da raiz
        if not os.path.exists(self.gesture_model_path):
            self.gesture_model_path = os.path.join("models", "modelo_gestos.pkl")
            self.hand_landmarker_path = os.path.join("models", "hand_landmarker.task")
            
        # Carregar modelo de gestos (Scikit-Learn)
        self.clf = joblib.load(self.gesture_model_path)
        
        # Opções do MediaPipe
        self.base_options = mp.tasks.BaseOptions(model_asset_path=self.hand_landmarker_path)
        self.running_mode = mp.tasks.vision.RunningMode.VIDEO
        
        self.options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=self.base_options,
            running_mode=self.running_mode,
            num_hands=2,
            min_hand_detection_confidence=0.7
        )
        
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(self.options)
