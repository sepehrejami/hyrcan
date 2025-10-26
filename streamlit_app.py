import io
import os
import yaml
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd

from src.face_service import FaceService
from src.gallery import Gallery
from src.recognizer import Recognizer
from src.responder import welcome_message
from src.attendance import AttendanceStore  


# ---------- Helpers ----------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """PIL -> OpenCV BGR"""
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def draw_box_and_label(frame_bgr: np.ndarray, box: dict, label: str):
    if not box:
        return frame_bgr
    x, y = int(box.get("x", 0)), int(box.get("y", 0))
    w, h = int(box.get("w", 0)), int(box.get("h", 0))
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        frame_bgr, label, (x, max(0, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )
    return frame_bgr


def to_rgb_for_streamlit(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# ---------- App Boot ----------
st.set_page_config(page_title="Smart Attendance (Streamlit)", page_icon="👤", layout="wide")

# Load config once (safe)
if "cfg" not in st.session_state:
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r", encoding="utf-8") as f:
            st.session_state.cfg = yaml.safe_load(f) or {}
    else:
        st.session_state.cfg = {}

cfg = st.session_state.cfg

# Lazy-singletons for services
if "gallery" not in st.session_state:
    st.session_state.gallery = Gallery(cfg.get("gallery_path", "data/gallery.json"))

if "face_service" not in st.session_state:
    st.session_state.face_service = FaceService(
        model_name=cfg.get("model_name", "Facenet"),
        min_face_size=int(cfg.get("min_face_size", 80)),
    )

if "recognizer" not in st.session_state:
    st.session_state.recognizer = Recognizer(
        st.session_state.gallery,
        threshold=float(cfg.get("recognition_threshold", 0.35)),
        metric=str(cfg.get("distance_metric", "cosine")),
    )

if "attendance" not in st.session_state:
    st.session_state.attendance = AttendanceStore(
        csv_path=cfg.get("attendance_csv", "data/attendance.csv"),
        daily_cooldown=bool(cfg.get("daily_cooldown", True)),
    )

# Keep a slot for last embedding across reruns
if "last_emb" not in st.session_state:
    st.session_state.last_emb = None

gallery: Gallery = st.session_state.gallery
face_service: FaceService = st.session_state.face_service
recognizer: Recognizer = st.session_state.recognizer
attendance: AttendanceStore = st.session_state.attendance

# UI: Sidebar
st.sidebar.header("Settings ⚙️")
new_threshold = st.sidebar.slider(
    "Recognition threshold",
    0.05, 1.0, float(recognizer.threshold), 0.01
)
new_metric = st.sidebar.selectbox(
    "Distance metric",
    ["cosine", "euclidean"],
    index=0 if recognizer.metric == "cosine" else 1
)

if (new_threshold != recognizer.threshold) or (new_metric != recognizer.metric):
    recognizer.threshold = float(new_threshold)
    recognizer.metric = new_metric
    st.sidebar.success("Recognizer settings updated.")

with st.sidebar.expander("Gallery", expanded=False):
    if len(gallery.people) == 0:
        st.info("Gallery is empty.")
    else:
        for name, embs in gallery.people.items():
            st.write(f"**{name}** — {len(embs)} embedding(s)")
    if st.button("Clear gallery ❌"):
        gallery.people = {}
        gallery.save()
        st.warning("Gallery cleared.")

with st.sidebar.expander("About", expanded=False):
    st.markdown(
        "- DeepFace for detection & embeddings\n"
        "- Cosine/Euclidean matching\n"
        "- CSV attendance with daily cooldown"
    )

# Main Layout
left, right = st.columns([2, 1])

with left:
    st.title("Smart Attendance — Streamlit Demo")
    st.caption("وب‌کم مرورگر → تشخیص چهره → شناسایی → افزودن فرد جدید + ثبت حضور")

    # Camera input (snapshot mode)
    shot = st.camera_input(
        "وب‌کم را روشن کن و یک عکس بگیر",
        help="پس از گرفتن عکس، پایین نتایج نمایش داده می‌شود."
    )

    messages = []
    frame_bgr = None

    if shot is not None:
        # Read frame
        pil_img = Image.open(io.BytesIO(shot.getvalue()))
        frame_bgr = pil_to_bgr(pil_img)

        # Detect & embed
        faces = face_service.detect_and_embed(frame_bgr)

        # Draw + recognize
        for emb, box in faces:
            name, score = recognizer.match(emb)
            if name:
                msg = welcome_message(name=name, known=True)
                label = f"{name} ({score:.3f})"
                # ثبت حضور (روزانه یک بار)
                wrote = attendance.mark(name, score)
                if wrote:
                    st.toast(f"Attendance marked for {name} ✅", icon="✅")
            else:
                msg = welcome_message(known=False)
                label = "Unknown"

            # Draw
            frame_bgr = draw_box_and_label(frame_bgr, box, label)
            messages.append((msg, name, score))

            # ذخیره‌ی امن امبدینگ در سشن برای استفاده بعد از rerun (دکمه Add)
            try:
                st.session_state["last_emb"] = [float(x) for x in emb]
            except Exception:
                # اگر نوع float32/ndarray بود و به float cast نشد، به همان صورت نگه می‌داریم
                st.session_state["last_emb"] = emb

    # Show image with boxes  (IMPORTANT: no string width; use `use_column_width`)
    if frame_bgr is not None:
        img = to_rgb_for_streamlit(frame_bgr)
        st.image(img, caption="Result", use_column_width="auto")  # "auto" | True | False

    # Messages
    st.subheader("Detections")
    if messages:
        for msg, name, score in messages:
            if name:
                st.success(f"{msg}  \nScore: **{score:.3f}**")
            else:
                st.info(msg)
    else:
        st.info("هنوز چهره‌ای تشخیص داده نشده یا عکسی نگرفتید.")

    st.divider()
    st.subheader("Add to Gallery")
    new_name = st.text_input("Name for the detected face", placeholder="e.g., Sara")
    add_btn = st.button("Add")

    if add_btn:
        emb = st.session_state.get("last_emb")
        if not new_name.strip():
            st.error("نام را وارد کن.")
        elif emb is None:
            st.error("هیچ امبدینگی در حافظه نیست. یک عکس تازه بگیر، بعد دوباره Add بزن.")
        else:
            gallery.add_person(new_name.strip(), emb)
            st.success(f"'{new_name.strip()}' added to gallery.")

with right:
    st.header("Attendance Report")
    rows = attendance.read_all()
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="attendance.csv", mime="text/csv")
    else:
        st.info("No attendance records yet.")

    st.markdown("---")
    st.header("How it works")
    st.markdown(
        """
        1) با `st.camera_input` یک عکس می‌گیری.  
        2) **DeepFace** تشخیص و embedding می‌سازد.  
        3) با **Recognizer** امبدینگ با گالری مقایسه می‌شود (Cosine/Euclidean).  
        4) اگر اسکُر از آستانه کمتر بود → *Known* و پیام خوشامد + ثبت حضور در CSV.  
        5) اگر Unknown بود، پایین اسم بده و **Add** کن تا به گالری اضافه شود.
        """
    )
    st.info("برای دقت بهتر: نور مناسب، چهره نزدیک به دوربین، و چند امبدینگ از هر نفر در شرایط مختلف ثبت کن.")
