import cv2
import numpy as np

class FrameProcessor:
    """Extração de gestos e desenho de rótulos (labels)."""
    
    @staticmethod
    def extract_features(hand_landmarks):
        """Converte landmarks para formato de features."""
        features = []
        for lm in hand_landmarks:
            features.extend([lm.x, lm.y, lm.z])
        return np.array(features).reshape(1, -1)

    @staticmethod
    def draw_hand_landmarks(frame, hand_landmarks):
        """Desenha conexões e pontos da mão."""
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),             # Polegar
            (0, 5), (5, 6), (6, 7), (7, 8),             # Indicador
            (5, 9), (9, 10), (10, 11), (11, 12),       # Médio
            (9, 13), (13, 14), (14, 15), (15, 16),     # Anelar
            (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Mínimo e base
        ]
        h, w, _ = frame.shape
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            sp = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
            ep = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
            cv2.line(frame, sp, ep, (255, 255, 255), 2)
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    def process_frame(self, frame, results, clf):
        """Adiciona rótulos de gestos e mãos no frame."""
        h, w, _ = frame.shape
        if results.hand_landmarks:
            for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                # Desenhar
                self.draw_hand_landmarks(frame, hand_landmarks)
                
                # Predição
                feat = self.extract_features(hand_landmarks)
                prediction = clf.predict(feat)[0]
                
                side = "Direita" if handedness[0].category_name == "Right" else "Esquerda"
                wrist_x = int(hand_landmarks[0].x * w)
                wrist_y = int(hand_landmarks[0].y * h)
                
                # Label no Pulso
                display_text = f"{side}: {prediction}"
                cv2.putText(frame, display_text, (wrist_x -50, wrist_y + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def get_detections(self, frame, results, clf):
        """Retorna uma lista de dicionários com os gestos detectados (sem desenhar)."""
        h, w, _ = frame.shape
        detections = []
        if results.hand_landmarks:
            for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                feat = self.extract_features(hand_landmarks)
                prediction = clf.predict(feat)[0]
                side = "Direita" if handedness[0].category_name == "Right" else "Esquerda"
                
                # Coordenadas do pulso
                wrist_x = int(hand_landmarks[0].x * w)
                wrist_y = int(hand_landmarks[0].y * h)
                
                detections.append({
                    "side": side,
                    "prediction": prediction,
                    "x": wrist_x,
                    "y": wrist_y
                })
        return detections
