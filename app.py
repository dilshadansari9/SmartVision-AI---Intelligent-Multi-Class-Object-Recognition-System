# smartvision_app.py
import streamlit as st
import os
import numpy as np
from PIL import Image

# ---------------- TensorFlow / Keras ----------------
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

# ---------------- YOLOv8 ----------------
from ultralytics import YOLO

# ---------------- App Config ----------------
st.set_page_config(page_title="SmartVision AI", layout="wide")
st.title("SmartVision AI")

# ---------------- Relative Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_BASE = os.path.join(BASE_DIR, "smartvision_dataset/classification")
TEST_DIR = os.path.join(DATASET_BASE, "test")

# ---------------- Fetch Class Names from Test Dir ----------------
class_names = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])
st.sidebar.write(f"Detected {len(class_names)} classes from test directory")

# ---------------- Model Info ----------------
models_info = {
    "EfficientNet": {
        "path": os.path.join(MODEL_DIR, "efficientnet_stage1.keras"),
        "preprocess": eff_preprocess
    },
    "MobileNetV2": {
        "path": os.path.join(MODEL_DIR, "mobilenetv2_smartvision.keras"),
        "preprocess": mob_preprocess
    },
    "VGG16": {
        "path": os.path.join(MODEL_DIR, "vgg16_best_model.h5"),
        "preprocess": vgg_preprocess
    }
}

# YOLOv8 path
yolo_model_path = os.path.join(MODEL_DIR, "yolov8m.pt")

# ---------------- Load Functions ----------------
@st.cache_resource
def load_classification_model(model_name):
    info = models_info[model_name]
    model = load_model(info["path"])
    preprocess_input = info["preprocess"]
    return model, preprocess_input

@st.cache_resource
def load_yolo_model():
    return YOLO(yolo_model_path)

# ---------------- Sidebar ----------------
page = st.sidebar.selectbox("Choose Page", ["Classification", "Object Detection (YOLOv8)"])

# ---------------- Classification Page ----------------
if page == "Classification":
    st.header("Image Classification")

    model_choice = st.selectbox("Select Classification Model", list(models_info.keys()))
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        model, preprocess_input = load_classification_model(model_choice)

        # Resize dynamically
        input_size = model.input_shape[1:3]
        img_array = image.img_to_array(img.resize(input_size))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        pred = model.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(pred)]
        confidence = np.max(pred) * 100

        st.success(f"Predicted Class: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

# ---------------- YOLOv8 Object Detection Page ----------------
# ---------------- YOLOv8 Object Detection Page ----------------
else:
    st.header("Object Detection (YOLOv8)")

    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

    uploaded_file = st.file_uploader(
        "Upload an image for YOLOv8",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Detecting objects..."):
            yolo_model = load_yolo_model()

            # YOLO inference
            results = yolo_model.predict(
                img,
                imgsz=640,
                conf=conf_threshold,
                device=0  # force GPU if available
            )

        # Plot detections
        annotated_img = results[0].plot()
        st.image(
            annotated_img,
            caption="YOLOv8 Detection Result",
            use_column_width=True
        )

        # ---- Detection summary table ----
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            st.subheader("Detected Objects")

            data = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = yolo_model.names[cls_id]

                data.append({
                    "Class": label,
                    "Confidence (%)": round(conf * 100, 2)
                })

            st.dataframe(data, use_container_width=True)
        else:
            st.warning("No objects detected.")
