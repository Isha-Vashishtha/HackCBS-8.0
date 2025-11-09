# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import regionprops, label, shannon_entropy
from PIL import Image
import pandas as pd
import os
from lime_explainer import explain_with_lime
import gdown

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.set_page_config(page_title="XAI Explorer — Leukemia Cell Classification", layout="wide")
st.title("📊 XAI Explorer — Leukemia Cell Classification")
st.markdown(
    "Choose a pre-trained model (VGG16 / ResNet50), upload a cell image, "
    "and visualize the most influential regions with explanations."
)

# --------------------------------
# 1️⃣ Model Selection (Google Drive IDs)
# --------------------------------
model_choices = {
    "VGG16": "1JwpNMwkvTeI8y1pC_LexEXurVIHWBNt8",  # Replace with your actual Drive ID
    "ResNet50": "15yqATv0VEb_tKNBbjsDpAqw7u-MG86od"
}

model_local_paths = {
    "VGG16": "models/vgg16_model",
    "ResNet50": "models/resnet50_model"
}

os.makedirs("models", exist_ok=True)

# --------------------------------
# Sidebar Configuration
# --------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox("Select Model", list(model_choices.keys()))
    xai_method = st.radio("Select XAI Method", ["Grad-CAM", "LIME"])
    threshold = st.slider("Heatmap Threshold", 100, 255, 180)
    class_labels = st.text_area(
        "Class Labels (comma-separated)",
        value="basophil,eosinophil,lymphocyte,monocyte,neutrophil"
    )

labels = [s.strip() for s in class_labels.split(",") if s.strip()]

# --------------------------------
# Download and Load Selected Model
# --------------------------------
@st.cache_resource
def download_and_load_model(drive_id, local_path):
    if not os.path.exists(local_path):
        st.info(f"Downloading model to {local_path} ...")
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, f"{local_path}.zip", quiet=False)
        # Unzip if needed
        import zipfile
        with zipfile.ZipFile(f"{local_path}.zip", 'r') as zip_ref:
            zip_ref.extractall(local_path)
    try:
        model = tf.keras.models.load_model(local_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = download_and_load_model(model_choices[selected_model], model_local_paths[selected_model])
if model is not None:
    st.sidebar.success(f"✅ Loaded model: {selected_model}")
else:
    st.stop()

# --------------------------------
# 2️⃣ Image Upload Section
# --------------------------------
st.header("🩸 Upload Leukemia Cell Image")
uploaded_img = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_img is not None and model is not None:
    pil_img = Image.open(uploaded_img).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_column_width=False, width=300)

    # --------------------------
    # XAI Method Selection
    # --------------------------
    if xai_method == "Grad-CAM":
        # Preprocess image
        img_size = (224, 224)
        img_resized = pil_img.resize(img_size)
        x = np.expand_dims(np.array(img_resized).astype(np.float32), axis=0)
        x_pre = tf.keras.applications.vgg16.preprocess_input(x)

        # Prediction
        preds = model.predict(x_pre)
        preds = np.squeeze(preds)
        pred_class = int(np.argmax(preds))
        confidence = float(preds[pred_class]) * 100

        if len(labels) != len(preds):
            labels = [f"class_{i}" for i in range(len(preds))]

        pred_label = labels[pred_class]
        st.markdown(f"### ✅ Prediction: **{pred_label}**  ({confidence:.2f}%)")

        # Grad-CAM Computation
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

        if last_conv_layer_name is None:
            st.error("No Conv2D layer found in model.")
        else:
            grad_model = tf.keras.models.Model(
                [model.inputs],
                [model.get_layer(last_conv_layer_name).output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(x_pre)
                loss = predictions[:, pred_class]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)

            # Resize and overlay
            img_original = np.array(pil_img)
            heatmap_resized = cv2.resize(np.array(heatmap), (img_original.shape[1], img_original.shape[0]))
            heatmap_255 = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_255, cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(img_original, 0.6, heatmap_color, 0.4, 0)

            st.subheader("🔥 Grad-CAM Heatmap")
            st.image(superimposed, width=400)

            # Region Feature Extraction
            _, thresh_img = cv2.threshold(heatmap_255, threshold, 255, cv2.THRESH_BINARY)
            labeled = label(thresh_img)
            regions = regionprops(labeled)

            feats = []
            for i, r in enumerate(regions):
                minr, minc, maxr, maxc = r.bbox
                centroid = r.centroid
                area = r.area
                solidity = r.solidity
                eccentricity = r.eccentricity
                patch = heatmap_resized[
                    max(int(centroid[0]) - 8, 0):min(int(centroid[0]) + 8, heatmap_resized.shape[0]),
                    max(int(centroid[1]) - 8, 0):min(int(centroid[1]) + 8, heatmap_resized.shape[1])
                ]
                entropy = float(shannon_entropy(patch)) if patch.size > 0 else 0.0
                score = float(np.mean(heatmap_resized[minr:maxr, minc:maxc]))
                feats.append({
                    "region_id": i+1, "area": area, "solidity": solidity,
                    "eccentricity": eccentricity, "entropy": entropy, "score": score,
                    "minr": minr, "minc": minc, "maxr": maxr, "maxc": maxc
                })

            if len(feats) == 0:
                st.warning("No activated regions detected. Try lowering the threshold.")
            else:
                df = pd.DataFrame(feats)
                st.subheader("📈 Region Feature Table")
                st.dataframe(df)

                most = df.loc[df['score'].idxmax()]

                highlighted = img_original.copy()
                cv2.rectangle(highlighted, (int(most['minc']), int(most['minr'])),
                            (int(most['maxc']), int(most['maxr'])), (0, 255, 0), 3)
                st.image(highlighted, caption="Most Influential Region", width=400)

    elif xai_method == "LIME":
        # LIME Explanation
        pred_label, confidence, lime_img_bound = explain_with_lime(model, pil_img, labels)
        st.markdown(f"### ✅ Prediction: **{pred_label}**  ({confidence:.2f}%)")
        st.subheader("🌈 LIME Explanation")
        st.image(lime_img_bound, use_column_width=True)
