import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

IMG_SIZE = 24
BATCH_SIZE = 32
EPOCHS = 20

def fine_grained_prune(weight_tensor: np.ndarray, sparsity: float) -> np.ndarray:
    """
    Magnitude-based pruning for a single weight tensor (NumPy array).
    :param weight_tensor: np.ndarray, weights from a Conv2D or Dense layer
    :param sparsity: float, percentage of zeros desired (0.0 to 1.0)
    :return: np.ndarray, the binary mask used for pruning
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    
    if sparsity == 1.0:
        weight_tensor.fill(0)
        return np.zeros_like(weight_tensor)
    elif sparsity == 0.0:
        return np.ones_like(weight_tensor)

    num_elements = weight_tensor.size
    
    # Step 1: Calculate the # of zeros
    num_zeros = int(round(num_elements * sparsity))
    
    # Step 2: Calculate importance (absolute value)
    importance = np.abs(weight_tensor)
    
    # Step 3: Calculate the pruning threshold
    # We flatten the array and find the value at the 'num_zeros' position
    flat_importance = importance.flatten()
    # np.partition is more efficient than full sorting
    threshold = np.partition(flat_importance, num_zeros - 1)[num_zeros - 1]
    
    # Step 4: Get binary mask (1 for nonzeros, 0 for zeros)
    # Weights strictly greater than the threshold stay
    mask = (importance > threshold).astype(np.float32)
    
    # Step 5: Apply mask to the weight tensor (in-place)
    weight_tensor[:] *= mask

    return mask

def prune_model(model, sparsity):
    for layer in model.layers:
        # Check if layer has weights (ignore MaxPooling, Flatten, etc.)
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            weights, biases = layer.get_weights()
            
            # Prune the weights (biases are usually left alone)
            mask = fine_grained_prune(weights, sparsity)
            
            # Update the layer with pruned weights
            layer.set_weights([weights, biases])
            
            print(f"Layer {layer.name} pruned to {sparsity*100}% sparsity.")

def representative_dataset():
    for images, _ in train_ds.take(100):
        for i in range(images.shape[0]):
            img = tf.cast(images[i:i+1], tf.float32)
            yield [img]

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Load dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "cew",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "cew",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

# Preftch and cache for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# # Normalize
# normalization_layer = layers.Rescaling(1./255)

# Build small CNN
model = models.Sequential([
    
    layers.Input(shape=(24,24,3)),
    # normalization_layer,
    
    layers.Conv2D(16, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save("based_model.keras")

################################# Mag-based Pruning - Pruning Ratio = 0.5 #################################
print("Start pruning...")
prune_model(model, 0.5)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save("pruned_model_TEST.keras")

# for layer in model.layers:
#     if hasattr(layer, 'kernel'):
#         weights = layer.get_weights()[0]
#         print(layer.name, "sparsity:",
#               (weights == 0).sum() / weights.size)

################################# INT8 Quantization #################################
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("eye_model.tflite", "wb") as f:
    f.write(tflite_model)