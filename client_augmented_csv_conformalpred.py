import numpy as np
import tensorflow as tf
import flwr as fl
import os
import csv
import sys
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
PORT       = "7518"
GPU_MEM_PER_CLIENT = 8000 # Measured in megabytes

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

        # Conformal Prediction stuff
        y_pred = self.model.predict(self.x_test)
        y_pred_confidence = np.max(y_pred, axis=1)

        # Calculate conformal prediction metrics
        residuals = 1 - y_pred_confidence
        conformal_interval_width = 2 * np.quantile(residuals, config["conformal_quantile"])
        coverage_rate = np.mean(residuals <= config["conformal_threshold"])

        metrics = {
            "conformal_interval_width": float(conformal_interval_width),
            "coverage_rate": float(coverage_rate),
        }

        print("\n\n")
        print("Conformal Interval Width: ", float(conformal_interval_width))
        print("Coverage Rate: ", float(coverage_rate))

        return self.model.get_weights(), len(self.x_train), metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy, auc, prec, rec = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": float(accuracy), "auc":float(auc), "precision":float(prec), "recall":float(rec)}

# Returns the model + a reference to base model (usefull for freezing)
def gen_resnet50_model():
  inputs = tf.keras.Input((150,150,3))
  model = tf.keras.applications.resnet.ResNet50(input_tensor=inputs, classes=8, weights=None)
  return tf.keras.Model(inputs, model.output), model

def gen_effnetb0_model():
  inputs = tf.keras.Input((150,150,3))
  model = tf.keras.applications.EfficientNetB0(input_tensor=inputs, classes=8, weights=None)
  return tf.keras.Model(inputs, model.output), model

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

def data_augmentation(images, labels):
    augmented_images = []
    augmented_labels = []
    
    for image, label in zip(images, labels):
        augmented_images.append(image)
        augmented_labels.append(label)
        
        # Horizontal Flip
        augmented_images.append(np.fliplr(image))
        augmented_labels.append(label)

        # Vertical Flip
        augmented_images.append(np.flipud(image))
        augmented_labels.append(label)
        
        # Random Rotation
        random_degree = np.random.uniform(-25, 25)
        augmented_images.append(transform.rotate(image, random_degree, mode='edge'))
        augmented_labels.append(label)
        
        # Random Translation
        random_x = np.random.uniform(-10, 10)
        random_y = np.random.uniform(-10, 10)
        translation_matrix = np.array([[1, 0, random_x], [0, 1, random_y], [0, 0, 1]])
        augmented_images.append(transform.warp(image, translation_matrix))
        augmented_labels.append(label)
    
    # Shuffling data
    combined = list(zip(augmented_images, augmented_labels))
    shuffle(combined)
    augmented_images[:], augmented_labels[:] = zip(*combined)
    
    return np.array(augmented_images), np.array(augmented_labels)

# Reading Data
print("Reading Data...")

x_train, y_train = load_images_from_csv(sys.argv[1])
x_test, y_test = load_images_from_csv(sys.argv[2])

print("Finished Reading Data!")

# Data Augmentation
print("Augmenting Data...")
x_train, y_train = data_augmentation(x_train, y_train)
print("Finished Augmenting Data!")

#model, base_model = gen_resnet50_model()
model, base_model = gen_effnetb0_model()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                             loss="categorical_crossentropy",
                             metrics=["accuracy",
                                      tf.keras.metrics.AUC(),
                                      tf.keras.metrics.Precision(),
                                      tf.keras.metrics.Recall()])
# Starting Client

print("Starting client...\n\n")
fl.client.start_numpy_client(server_address=f"{HOST}:{PORT}", client=FlowerClient(x_train, y_train, x_test, y_test, model, base_model))
print("\n\nTraining Done!")
