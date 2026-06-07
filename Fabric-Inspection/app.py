import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# -----------------------------

# PAGE SETTINGS

# -----------------------------

st.set_page_config(
page_title="AI Fabric Inspection",
page_icon="🧵",
layout="wide"
)

# -----------------------------

# CUSTOM CSS

# -----------------------------

st.markdown("""

<style>
.stApp {
    background-color: #0f172a;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 25px;
}

.status-ok {
    background-color: #14532d;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.status-ng {
    background-color: #7f1d1d;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
</style>

""", unsafe_allow_html=True)

# -----------------------------

# LOAD MODEL

# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(**file**))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

if not os.path.exists(MODEL_PATH):
st.error(f"Model not found: {MODEL_PATH}")
st.stop()

try:
model = YOLO(MODEL_PATH)
except Exception as e:
st.error(f"Failed to load model: {e}")
st.stop()

# -----------------------------

# HEADER

# -----------------------------

st.markdown(
'<p class="main-title">🧵 AI Fabric Quality Inspection System</p>',
unsafe_allow_html=True
)

st.markdown(
'<p class="sub-title">Upload a fabric image and detect defects using YOLOv8</p>',
unsafe_allow_html=True
)

# -----------------------------

# FILE UPLOAD

# -----------------------------

uploaded_file = st.file_uploader(
"Upload Fabric Image",
type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

```
image = Image.open(uploaded_file).convert("RGB")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    image.save(tmp.name, format="JPEG")

try:
    results = model(tmp.name)
except Exception as e:
    st.error(f"Inference Error: {e}")
    st.stop()

annotated_image = results[0].plot()

with col2:
    st.subheader("Detection Result")
    st.image(annotated_image, use_container_width=True)

st.markdown("---")

boxes = results[0].boxes

if len(boxes) > 0:

    st.markdown(
        '<div class="status-ng">❌ NG PRODUCT DETECTED</div>',
        unsafe_allow_html=True
    )

    st.write("### Defects Found")

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        st.write(
            f"• {model.names[cls_id]} | Confidence: {conf:.2f}"
        )

else:

    st.markdown(
        '<div class="status-ok">✅ FABRIC PASSED INSPECTION</div>',
        unsafe_allow_html=True
    )

    st.balloons()
```

st.markdown("---")
st.caption("Developed with Streamlit + YOLOv8")
