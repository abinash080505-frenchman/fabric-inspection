import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# --------------------------------------------------

# PAGE SETTINGS

# --------------------------------------------------

st.set_page_config(
page_title="AI Fabric Inspection",
page_icon="🧵",
layout="wide"
)

# --------------------------------------------------

# CUSTOM CSS

# --------------------------------------------------

st.markdown("""

<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.main-title {
    text-align:center;
    color:white;
    font-size:42px;
    font-weight:bold;
}

.sub-title {
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
    margin-bottom:30px;
}

.block {
    background-color:#111827;
    padding:20px;
    border-radius:15px;
    border:1px solid #374151;
}

.status-ok {
    background:#065f46;
    color:white;
    padding:15px;
    border-radius:10px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

.status-ng {
    background:#991b1b;
    color:white;
    padding:15px;
    border-radius:10px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}
</style>

""", unsafe_allow_html=True)

# --------------------------------------------------

# MODEL LOADING

# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

if not os.path.exists(MODEL_PATH):
st.error(f"Model file not found: {MODEL_PATH}")
st.stop()

try:
model = YOLO(MODEL_PATH)
except Exception as e:
st.error(f"Error loading model: {e}")
st.stop()

# --------------------------------------------------

# HEADER

# --------------------------------------------------

st.markdown(
'<div class="main-title">🧵 AI Fabric Quality Inspection System</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="sub-title">Upload a fabric image and detect defects using YOLOv8</div>',
unsafe_allow_html=True
)

# --------------------------------------------------

# FILE UPLOADER

# --------------------------------------------------

uploaded_file = st.file_uploader(
"📤 Upload Fabric Image",
type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------

# PREDICTION

# --------------------------------------------------

if uploaded_file is not None:

```
image = Image.open(uploaded_file).convert("RGB")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Original Image")
    st.image(image, use_container_width=True)

with st.spinner("Inspecting Fabric..."):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name, format="JPEG")
        results = model(tmp.name)

annotated_image = results[0].plot()

with col2:
    st.subheader("🎯 Detection Result")
    st.image(annotated_image, use_container_width=True)

st.markdown("---")

boxes = results[0].boxes

if len(boxes) > 0:

    st.markdown(
        '<div class="status-ng">❌ NG PRODUCT DETECTED</div>',
        unsafe_allow_html=True
    )

    st.metric("Detected Defects", len(boxes))

    st.subheader("Defect Report")

    for box in boxes:
        defect = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])

        st.write(
            f"• **{defect}** — Confidence: **{confidence:.2%}**"
        )

else:

    st.markdown(
        '<div class="status-ok">✅ FABRIC PASSED INSPECTION</div>',
        unsafe_allow_html=True
    )

    st.metric("Detected Defects", 0)
```

# --------------------------------------------------

# FOOTER

# --------------------------------------------------

st.markdown("---")
st.caption(
"Powered by YOLOv8 • Fabric Defect Detection System"
)
