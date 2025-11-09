# lime_explainer.py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from lime import lime_image
from skimage.segmentation import mark_boundaries

def explain_with_lime(model, pil_img, class_labels):
    """
    Returns predicted label, confidence, and LIME explanation image.
    """

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

    # Adjust label list if mismatch
    if len(class_labels) != len(preds):
        class_labels = [f"class_{i}" for i in range(len(preds))]

    pred_label = class_labels[pred_class]

    # -----------------------------
    # LIME Image Explanation
    # -----------------------------
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(img_resized).astype(np.double),
        classifier_fn=lambda imgs: model.predict(tf.keras.applications.vgg16.preprocess_input(imgs)),
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    lime_img, mask = explanation.get_image_and_mask(
        pred_class,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    lime_img_bound = mark_boundaries(lime_img / 255.0, mask)

    return pred_label, confidence, lime_img_bound
