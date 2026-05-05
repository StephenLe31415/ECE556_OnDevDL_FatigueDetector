import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ================= CONFIG =================
IMG_SIZE   = 24
BATCH_SIZE = 32
EPOCHS     = 20
DATA_DIR   = "yawn_dataset"   # yawn/, no yawn/

# ==========================================

def load_images(folder, label):
    images, labels = [], []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(label)
    return images, labels


# ================= LOAD =================
yawn_X,    yawn_y    = load_images(os.path.join(DATA_DIR, "yawn"),    0)
no_yawn_X, no_yawn_y = load_images(os.path.join(DATA_DIR, "no yawn"), 1)

X     = np.array(yawn_X + no_yawn_X, dtype=np.float32) / 255.0
y_int = np.array(yawn_y + no_yawn_y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_int, test_size=0.2, stratify=y_int, random_state=42
)


# ================= PRUNING =================
def fine_grained_prune(weight_tensor: np.ndarray, sparsity: float) -> np.ndarray:
    sparsity = min(max(0.0, sparsity), 1.0)

    if sparsity == 1.0:
        weight_tensor.fill(0)
        return np.zeros_like(weight_tensor)
    elif sparsity == 0.0:
        return np.ones_like(weight_tensor)

    num_zeros   = int(round(weight_tensor.size * sparsity))
    importance  = np.abs(weight_tensor)
    flat        = importance.flatten()
    threshold   = np.partition(flat, num_zeros - 1)[num_zeros - 1]
    mask        = (importance > threshold).astype(np.float32)
    weight_tensor[:] *= mask
    return mask


def prune_model(model, sparsity):
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            weights, biases = layer.get_weights()
            fine_grained_prune(weights, sparsity)
            layer.set_weights([weights, biases])
            print(f"  {layer.name} pruned to {sparsity*100:.0f}% sparsity")


# ================= MODEL =================
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= TRAIN =================
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

model.save("mouth_base_model.keras")
print("Base mouth model saved.")

# ================= PRUNING + FINE-TUNE =================
print("\nPruning model...")
prune_model(model, 0.5)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

model.save("mouth_pruned_model.keras")
print("Pruned mouth model saved.")

# ================= INT8 =================
def rep_data():
    for i in range(100):
        yield [np.expand_dims(X_train[i], axis=0).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations             = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset    = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type      = tf.int8
converter.inference_output_type     = tf.int8

tflite_model = converter.convert()

with open("mouth_model.tflite", "wb") as f:
    f.write(tflite_model)