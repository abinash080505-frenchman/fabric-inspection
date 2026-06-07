import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load the model you just trained
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

st.title("🧵 AI Fabric Quality Inspection System")
st.write("Upload a fabric image to detect defects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Temporary save for YOLO
image = image.convert("RGB")

with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    image.save(tmp.name, format="JPEG")

try:
    results = model(tmp.name)
except Exception as e:
    st.error(f"Inference Error: {e}")
    st.stop()

    # Show Result
    annotated_image = results[0].plot()
    st.subheader("🔍 Detection Result")
    st.image(annotated_image, use_container_width=True)

    if len(results[0].boxes) > 0:
        st.error(f"❌ Status: NG ({len(results[0].boxes)} defects found)")
        for box in results[0].boxes:
            st.write(f"- {model.names[int(box.cls[0])]} (Conf: {float(box.conf[0]):.2f})")
    else:
        st.success("✅ Status: OK")
