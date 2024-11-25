#
#
#
# ATTENTION! KERAS MUST BE AT 3.3.3 ! OTHER PACKAGES CAN REMAIN THE SAME.
#
#
#



import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.ops as ops
import flwr as fl
import os
import csv
import sys
import stainNorm_Macenko
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage import transform
from random import shuffle

if len(sys.argv) != 3:
    print(f"ERROR: Wrong usage!\nCorrect one: python3 {sys.argv[0]} 'train_file.csv' 'validation_file.csv' ")
    exit(1)

TESTING    = False
TRAIN_SIZE = 0.8
HOST       = "192.168.0.39"
PORT       = "7517"
GPU_MEM_PER_CLIENT = 11000 # Measured in megabytes

# GPU MEMORY GROWTH POLITICS

# Fixed Limit
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEM_PER_CLIENT)])

# Dynamic Limit
#physical_devices = tf.config.list_physical_devices('GPU')
#for phy_dev in physical_devices:
#  tf.config.experimental.set_memory_growth(phy_dev, True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_test, y_test, model, base_model_ref, lr=0.001, local_epochs=10, batch_size=128, verbose=1):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
        self.base_model_ref = base_model_ref
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Learning rate decay
        if bool(config["lr_decay"]):
          self.lr *= float(config["decay_factor"])
          self.model.compile(optimizer=tf.keras.optimizers.Adam(self.lr),
                             loss="categorical_crossentropy",
                             metrics=["accuracy",
                                      tf.keras.metrics.AUC(),
                                      tf.keras.metrics.Precision(),
                                      tf.keras.metrics.Recall()])

        # Freeze
        if bool(config["alter_trainable"]):
          self.base_model_ref.trainable = bool(config["trainable"])

        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=self.local_epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy, auc, prec, rec = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": float(accuracy), "auc":float(auc), "precision":float(prec), "recall":float(rec)}

def load_images_from_csv(csv_path):
    images = []
    labels = []

    header = True
    with open(csv_path) as f:
      reader = csv.reader(f)
      for row in reader:

        if header:
          header = False
          continue
        
        images.append(imread(row[0])) # Feature

        label = np.zeros((8,), dtype=np.int64) # Label
        label[int(row[1])] = 1
        labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Reading Data
print("Reading Data...")

num_classes = 8
input_shape = (150, 150, 3)
x_train, y_train = load_images_from_csv(sys.argv[1])
x_test, y_test = load_images_from_csv(sys.argv[2])

print("Finished Reading Data!")

# Macenko normalization
print("Starting Macenko Normalization...")

normalizer = stainNorm_Macenko.Normalizer()
normalizer.fit(x_train[0])

def norm(img):
  return normalizer.transform(img)

x_train = norm(x_train)
x_test = norm(x_test)

print("Finished Macenko Normalization!")

# Vision transformers stuff

# HyperParams

batch_size = 64
image_size = 150  # We'll resize input images to this size
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 12
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 12
mlp_head_units = [
    256
]  # Size of the dense layers of the final classifier

"""
## Use data augmentation
"""

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)

# Traditional NN
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Vision Transformer Patches
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

# Vision Transformer Patch Encoder
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config

def create_vit_classifier():
    inputs = keras.Input(shape=input_shape)
    # Using augmentation on image files themselves.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model, model

model, base_model = create_vit_classifier()
model.compile(optimizer=keras.optimizers.Adam(0.001),
                             loss="categorical_crossentropy",
                             metrics=["accuracy",
                                      tf.keras.metrics.AUC(),
                                      tf.keras.metrics.Precision(),
                                      tf.keras.metrics.Recall()])

# Starting Client

print("Starting client...\n\n")
fl.client.start_numpy_client(server_address=f"{HOST}:{PORT}", client=FlowerClient(x_train, y_train, x_test, y_test, model, base_model, local_epochs=20, batch_size=batch_size))
print("\n\nTraining Done!")
