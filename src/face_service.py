import cv2
from deepface import DeepFace

class FaceService:
    def __init__(self, model_name="Facenet", detector_backend="retinaface", min_face_size=80):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.min_face_size = min_face_size

    def detect_and_embed(self, frame_bgr):
        # DeepFace ورودی را می‌تواند آرایه‌ی RGB بگیرد
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        reps = DeepFace.represent(
            img_path=rgb,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            enforce_detection=False
        ) or []
        results = []
        for r in reps:
            emb = r.get("embedding")
            box = r.get("facial_area") or {}
            if emb is None or not box:
                continue
            w = box.get("w", 0); h = box.get("h", 0)
            if min(w, h) < self.min_face_size:
                continue
            results.append((emb, box))
        # بزرگ‌ترین چهره اول
        results.sort(key=lambda it: (it[1].get("w",0)*it[1].get("h",0)), reverse=True)
        return results
