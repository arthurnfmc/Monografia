import numpy as np
import tensorflow as tf
import flwr as fl
import os
import sys
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.io import imread

if len(sys.argv)!=3:
    print(f"ERROR: Wrong usage!\nCorrect one: python3 {sys.argv[0]} 'path_to_model.npz' 'test_file.csv' ")
    exit(1)

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

path = sys.argv[1]

print("Reading Data...")
x_test, y_test = load_images_from_csv(sys.argv[2])
print("Finished Reading Data!")

model, base_model = gen_effnetb0_model()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                             loss="categorical_crossentropy",
                             metrics=["accuracy",
                                      tf.keras.metrics.AUC(),
                                      tf.keras.metrics.Precision(),
                                      tf.keras.metrics.Recall()])

data = np.load(path)
weights = []
for key in data.keys():
    weights.append(data[key])
model.set_weights(weights)
loss, accuracy, auc, prec, rec = model.evaluate(x_test, y_test, verbose=1)

with open(os.path.join(os.path.dirname(path), "best.log"), "w+") as f:
  f.write(f"MODEL TESTED: {path}\n")
  f.write(f"LOSS = {loss}\nACC = {accuracy}\nAUC = {auc}\nPREC = {prec}\nREC = {rec}\n")

print(f"TEST DONE! RESULTS SAVED ON {os.path.join(os.path.dirname(path), 'best.log')}")
print("CALCULATING CONFUSION MATRIX...")

class_names = ['TUMOR', 'STROMA', 'COMPLEX', 'LYMPHO', 'DEBRIS', 'MUCOSA', 'ADIPOSE', 'EMPTY']

predictions = model.predict(x_test)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(true_classes, predicted_classes)

plt.ioff()

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predição')
plt.ylabel('Real')
plt.title('Matriz de Confusão')

plt.savefig(os.path.join(os.path.dirname(path), "confusion_matrix.png"))
plt.close()
print("CONFUSION MATRIX DONE!")
