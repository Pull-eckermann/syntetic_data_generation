import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
from sklearn import metrics


BATCH_SIZE = 32
IMG_SIZE = (160, 160)

export_path = "retrained/saved_models/UFPR04-real"

try:
    validation_dir = sys.argv[1]
except:
    print("Please specify the path to the dataset dir")
    exit(0)

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

AUTOTUNE = tf.data.AUTOTUNE
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# Restore the model
model = tf.keras.models.load_model(export_path)

#Predict
loss, accuracy = model.evaluate(validation_dataset)
print("Loss: {}".format(loss))
print("Accuracy: {}".format(accuracy))
"""
validation_labels = np.concatenate([y for _, y in validation_dataset], axis=0)
print(validation_labels)

predictions = model.predict(validation_dataset).flatten()
predicted_labels = tf.nn.sigmoid(predictions)
predicted_labels = tf.where(predicted_labels < 0.5, 0, 1)

print("Accuracy: ",metrics.accuracy_score(validation_labels, predicted_labels))

class_names = ['Empty', 'Occupied']
class_names = np.array(class_names)

scores = []
for image_batch, true_id in validation_dataset:
    predicted_batch = model.predict_on_batch(image_batch).flatten()
    predicted_ids = tf.nn.sigmoid(predicted_batch)
    predicted_ids = tf.where(predicted_ids < 0.5, 0, 1)
    scores.append(metrics.accuracy_score(true_id, predicted_ids))

print('Total accuracy: {}'.format(sum(scores) / len(scores)))
"""