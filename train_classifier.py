train_classifier.py
# ============================================================
# ✅ IMPORT LIBRARIES
# ============================================================
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os

# ============================================================
# ✅ SET PATH (change this to your dataset folder)
# ============================================================
data_dir = r"D:\XAI_Feature_Extraction\dataset"

# ============================================================
# ✅ DATA GENERATOR WITH VALIDATION SPLIT
# ============================================================
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2,  # 20% of data for validation
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = len(train_gen.class_indices)
print("\n✅ Classes detected:", train_gen.class_indices)

# ============================================================
# ✅ FUNCTION TO BUILD MODEL (COMMON)
# ============================================================
def build_model(base):
    base.trainable = False  # Freeze convolutional base
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ============================================================
# ✅1️⃣ TRAIN VGG16
# ============================================================
print("\n==============================")
print("🔵 TRAINING VGG16 MODEL")
print("==============================")

vgg_base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
vgg_model = build_model(vgg_base)

vgg_ckpt = ModelCheckpoint("vgg16_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
vgg_es = EarlyStopping(patience=5, monitor="val_accuracy", restore_best_weights=True, verbose=1)

vgg_history = vgg_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[vgg_ckpt, vgg_es]
)

# ============================================================
# ✅ 2️⃣ TRAIN RESNET50
# ============================================================
print("\n==============================")
print("🟣 TRAINING RESNET50 MODEL")
print("==============================")

resnet_base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
resnet_model = build_model(resnet_base)

resnet_ckpt = ModelCheckpoint("resnet50_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
resnet_es = EarlyStopping(patience=5, monitor="val_accuracy", restore_best_weights=True, verbose=1)

resnet_history = resnet_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[resnet_ckpt, resnet_es]
)

# ============================================================
# ✅ FINAL PRINT
# ============================================================
print("\n✅ Training complete!")
print("Saved models: vgg16_model.h5, resnet50_model.h5")
