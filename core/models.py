import os, joblib, mediapipe as mp

class ModelLoader:
    def __init__(self, models_dir=None):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not models_dir:
            models_dir = os.path.join(base_path, "models")
        
        self.hand_landmarker_path = os.path.join(models_dir, "hand_landmarker.task")
        self.gesture_model_path = os.path.join(models_dir, "modelo_gestos.pkl")
        
        # VERIFICAÇÃO RIGOROSA
        if not os.path.exists(self.hand_landmarker_path):
            print(f"DEBUG ERRO: hand_landmarker_path NAO EXISTE EM {self.hand_landmarker_path}")
            # Tenta um último recurso na raiz
            self.hand_landmarker_path = "models/hand_landmarker.task"
            self.gesture_model_path = "models/modelo_gestos.pkl"
            
        if not os.path.exists(self.hand_landmarker_path):
            raise FileNotFoundError(f"ARQUIVOS FALTANDO NO SERVIDOR: Verifique se subiu a pasta 'models' para o GitHub!")

        print(f"LOG: Carregando modelos de {self.hand_landmarker_path}...")
        self.clf = joblib.load(self.gesture_model_path)
        self.base_options = mp.tasks.BaseOptions(model_asset_path=self.hand_landmarker_path)
        self.options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=self.base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(self.options)
