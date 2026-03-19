import cv2, numpy as np

class FrameProcessor:
    @staticmethod
    def extract_features(hand_landmarks):
        features = []
        for lm in hand_landmarks:
            features.extend([lm.x, lm.y, lm.z])
        return np.array(features).reshape(1, -1)

    @staticmethod
    def draw_hand_landmarks(frame, hand_landmarks):
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
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
        h, w, _ = frame.shape
        if results.hand_landmarks:
            for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                self.draw_hand_landmarks(frame, hand_landmarks)
                feat = self.extract_features(hand_landmarks)
                prediction = clf.predict(feat)[0]
                side = "Direita" if handedness[0].category_name == "Right" else "Esquerda"
                wrist_x = int(hand_landmarks[0].x * w)
                wrist_y = int(hand_landmarks[0].y * h)
                cv2.putText(frame, f"{side}: {prediction}", (wrist_x - 50, wrist_y + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def get_detections(self, frame, results, clf):
        h, w, _ = frame.shape
        detections = []
        if results.hand_landmarks:
            for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                feat = self.extract_features(hand_landmarks)
                prediction = clf.predict(feat)[0]
                side = "Direita" if handedness[0].category_name == "Right" else "Esquerda"
                detections.append({
                    "side": side,
                    "prediction": prediction,
                    "x": int(hand_landmarks[0].x * w),
                    "y": int(hand_landmarks[0].y * h)
                })
        return detections
