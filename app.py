# app.py
import time
import yaml
import cv2
from statistics import mean

from src.face_service import FaceService
from src.gallery import Gallery
from src.recognizer import Recognizer
from src.responder import welcome_message

# -----------------------------
# Optional TTS (pyttsx3)
# -----------------------------
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except Exception:
    _TTS_AVAILABLE = False

_tts_engine = None
def say(text: str):
    """Speak text if TTS is available; otherwise do nothing."""
    global _tts_engine
    if not _TTS_AVAILABLE:
        return
    try:
        if _tts_engine is None:
            _tts_engine = pyttsx3.init()
            # Optional tuning:
            # _tts_engine.setProperty("rate", 175)
            # _tts_engine.setProperty("volume", 1.0)
        _tts_engine.say(text)
        _tts_engine.runAndWait()
    except Exception:
        # Fail silently if TTS breaks at runtime
        pass

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Load config
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    model_name = cfg.get("model_name", "Facenet")
    metric = cfg.get("distance_metric", "cosine")
    threshold = float(cfg.get("recognition_threshold", 0.35))
    min_face_size = int(cfg.get("min_face_size", 80))
    gallery_path = cfg.get("gallery_path", "data/gallery.json")

    # Services
    gallery = Gallery(gallery_path)
    face_service = FaceService(model_name=model_name, min_face_size=min_face_size)
    recognizer = Recognizer(gallery, threshold=threshold, metric=metric)

    # Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open default camera (index=0).")

    print("Press 'q' to quit. Press 'n' to name & save new face.")
    if _TTS_AVAILABLE:
        print("[TTS] pyttsx3 detected. Audio greetings enabled.")
    else:
        print("[TTS] pyttsx3 not installed. Run: pip install pyttsx3 (optional)")

    WINDOW = "Smart Attendance (MVP)"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    last_status = ""
    last_seen = {}     # name -> last timestamp
    unknown_last = 0.0 # cooldown for unknown
    COOLDOWN_SEC = 5

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = face_service.detect_and_embed(frame)

        for emb, box in faces:
            name, score = recognizer.match(emb)
            now = time.time()

            # Decide message + cooldown
            if name:
                # Known person
                if now - last_seen.get(name, 0) > COOLDOWN_SEC:
                    msg = welcome_message(name=name, known=True)
                    last_seen[name] = now
                    say(msg)
                else:
                    msg = f"Hello, {name}"
                status = f"{msg} (score={score:.3f})"
            else:
                # Unknown person
                msg = welcome_message(known=False)
                status = f"{msg} (new face)"
                if now - unknown_last > COOLDOWN_SEC:
                    unknown_last = now
                    say(msg)

            # Draw UI
            if box:
                x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame, msg, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            # Log (avoid spam)
            if status != last_status:
                print(status)
                last_status = status

        # Show frame
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('n'):
            # Register the largest/primary detected face with averaged embedding
            if faces:
                # Collect a small batch across frames to reduce noise
                batch = [faces[0][0]]  # start with current primary face embedding
                FRAMES_TO_ACCUM = 4
                for _ in range(FRAMES_TO_ACCUM):
                    ok2, fr2 = cap.read()
                    if not ok2:
                        break
                    det2 = face_service.detect_and_embed(fr2)
                    if det2:
                        batch.append(det2[0][0])

                try:
                    new_name = input("Enter name for this face: ").strip()
                except EOFError:
                    new_name = ""

                if new_name:
                    # Element-wise mean
                    mean_emb = [mean(vals) for vals in zip(*batch)]
                    gallery.add_person(new_name, mean_emb)
                    info = f"[INFO] Added '{new_name}' to gallery."
                    print(info)
                    say(f"Saved. Welcome, {new_name}.")
                else:
                    print("[WARN] Empty name. Skipped.")
            else:
                warn = "[WARN] No face detected to name."
                print(warn)
                say("No face detected to save.")

    cap.release()
    cv2.destroyAllWindows()
