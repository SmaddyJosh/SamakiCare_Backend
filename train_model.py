
#Fish Disease Detection — Training Script
#data discovery
#data augmentation
#handling imbalance
#model architecture efficient<net -image classification
#training the head first, then fine-tuning
#confusion matrix and training curves for evaluation

import os, json, shutil, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


import kagglehub
import uuid
#***data collection ***
datasets = [
    "utpoldas/freshwater-fish-disease-dataset",
    "subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia",
    "faizakram/fish-disease-image-datasets"
]

COMBINED_RAW_DIR = "combined_raw_data"
#***structure the data in a way that is easy to work with for training***
os.makedirs(COMBINED_RAW_DIR, exist_ok=True)

def find_image_root(base):
    for root, dirs, files in os.walk(base):
        images = [f for f in files if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if images:
            # Return parent of whichever folder contains images
            return os.path.dirname(root) if os.path.basename(root) != base else root
    return base

print(" Starting multi-dataset download and merge...")
for ds in datasets:
    try:
        print(f" Downloading dataset: {ds}")
        path = kagglehub.dataset_download(ds)
        ds_root = find_image_root(path)
        print(f" Image root found at: {ds_root}")
        #***Data split ***
        # Merge folders (classes) into COMBINED_RAW_DIR
        for d in os.listdir(ds_root):
            src_dir = os.path.join(ds_root, d)
            if os.path.isdir(src_dir):
             
                clean_class = d.lower().replace(" ", "_").replace("-", "_")
                if "healthy" in clean_class or "normal" in clean_class:
                    clean_class = "healthy"
                
                dest_dir = os.path.join(COMBINED_RAW_DIR, clean_class)
                os.makedirs(dest_dir, exist_ok=True)
                
                # Copy images over with a unique prefix to prevent overwriting
                for img in os.listdir(src_dir):
                    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                        new_name = f"{uuid.uuid4().hex[:8]}_{img}"
                        shutil.copy2(os.path.join(src_dir, img), os.path.join(dest_dir, new_name))
    except Exception as e:
        print(f" Could not process dataset {ds}: {e}")

raw_root = COMBINED_RAW_DIR
print(f" Data combined into: {raw_root}")

#**dta augementation and preparation for training**
classes = sorted([
    d for d in os.listdir(raw_root)
    if os.path.isdir(os.path.join(raw_root, d))
])
print(f" Classes found ({len(classes)}): {classes}")


SPLIT_DIR  = "fish_split"
VAL_RATIO  = 0.2
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 30
OUTPUT_DIR = "model_output"
SEED       = 42

random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for split in ("train", "val"):
    for cls in classes:
        os.makedirs(os.path.join(SPLIT_DIR, split, cls), exist_ok=True)

for cls in classes:
    src = os.path.join(raw_root, cls)
    #handle imbalance by ensuring at least one image goes to validation, even for small classes**
    imgs = [f for f in os.listdir(src) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * VAL_RATIO))
    for i, img in enumerate(imgs):
        split = "val" if i < n_val else "train"
        dst = os.path.join(SPLIT_DIR, split, cls, img)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(src, img), dst)

print("Train/val split ready.")


train_gen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.75, 1.25],
    fill_mode="nearest",
).flow_from_directory(
    os.path.join(SPLIT_DIR, "train"),
    target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=True, seed=SEED,
)

val_gen = ImageDataGenerator(rescale=1/255.).flow_from_directory(
    os.path.join(SPLIT_DIR, "val"),
    target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False,
)

NUM_CLASSES = train_gen.num_classes
CLASS_NAMES = list(train_gen.class_indices.keys())
print(f"Classes: {CLASS_NAMES}")

with open(os.path.join(OUTPUT_DIR, "class_names.json"), "w") as f:
    json.dump(CLASS_NAMES, f, indent=2)

# Class weights for imbalance
counts = np.bincount(train_gen.classes)
class_weights = {i: len(train_gen.classes)/(NUM_CLASSES*c) for i,c in enumerate(counts)}


base = MobileNetV2(weights="imagenet", include_top=False,
                      input_shape=(*IMAGE_SIZE, 3))
base.trainable = False

inp = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x   = base(inp, training=False)
x   = layers.GlobalAveragePooling2D()(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dense(512, activation="relu")(x)
x   = layers.Dropout(0.4)(x)
x   = layers.Dense(256, activation="relu")(x)
x   = layers.Dropout(0.3)(x)
out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)
model.summary()


ckpt = os.path.join(OUTPUT_DIR, "best_model.keras")
cbs  = [
    ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7),
    CSVLogger(os.path.join(OUTPUT_DIR, "phase1_log.csv")),
]
#we train new layers 
print("\n🚀 Phase 1: training head …")
h1 = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
               class_weight=class_weights, callbacks=cbs, verbose=2)


base.trainable = True
for layer in base.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)
cbs2 = [
    ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-8),
    CSVLogger(os.path.join(OUTPUT_DIR, "phase2_log.csv")),
]
#help see the fish better
print("\n🔓 Phase 2: fine-tuning …")
h2 = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
               class_weight=class_weights, callbacks=cbs2, verbose=2)


model.save(os.path.join(OUTPUT_DIR, "fish_disease_model.keras"))

# Also export as TF SavedModel (easier to serve)
model.export(os.path.join(OUTPUT_DIR, "saved_model"))
print(f"\nModel saved to {OUTPUT_DIR}/")

# Classification report
val_gen.reset()
preds = model.predict(val_gen, verbose=2)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(max(6, NUM_CLASSES), max(5, NUM_CLASSES-1)))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix"); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)

# Training curves
def mh(h, k): return h.history.get(k, [])
ep = list(range(len(mh(h1,"accuracy") + mh(h2,"accuracy"))))
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for ax, m, title in zip(axs, ["accuracy","loss"], ["Accuracy","Loss"]):
    ax.plot(mh(h1,m)+mh(h2,m), label=f"Train")
    ax.plot(mh(h1,f"val_{m}")+mh(h2,f"val_{m}"), "--", label="Val")
    ax.axvline(len(mh(h1,m))-1, color="gray", linestyle=":", alpha=.6, label="Fine-tune")
    ax.set_title(title); ax.legend(); ax.set_xlabel("Epoch")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
print("📊 Plots saved.")
