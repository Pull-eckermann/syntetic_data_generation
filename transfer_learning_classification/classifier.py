import numpy as np
import sys
import tensorflow as tf
#from sklearn import metrics

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

export_path = "retrained/saved_models/CNR-rrect-v3"

try:
    validation_dir = sys.argv[1]
except:
    print("Please specify the path to the dataset dir")
    exit(0)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(validation_dir,
                                                                 shuffle=False,
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
labels = []
for image_batch, true_id in validation_dataset:
    labels = labels + list(true_id)

predictions = model.predict(validation_dataset).flatten()
predicted_labels = tf.nn.sigmoid(predictions)
predicted_labels = tf.where(predicted_labels < 0.5, 0, 1)
predicted_labels = list(predicted_labels)

print("Total accuracy: ",metrics.accuracy_score(labels, predicted_labels))
matrix = tf.math.confusion_matrix(labels, predicted_labels)

print("Confusion Matrix:")
print(np.asarray(matrix))

----------------------------------------------
scores = []
labels = []
predictions = []
for image_batch, true_id in validation_dataset:
    predicted_batch = model.predict_on_batch(image_batch).flatten()
    predicted_ids = tf.nn.sigmoid(predicted_batch)
    predicted_ids = tf.where(predicted_ids < 0.5, 0, 1)
    scores.append(metrics.accuracy_score(true_id, predicted_ids))
    labels = labels + list(true_id)
    predictions = predictions + list(predicted_ids)

print('Total accuracy: {}'.format(sum(scores) / len(scores)))

matrix = tf.math.confusion_matrix(labels, predictions)

print("Confusion Matrix:")
print(np.asarray(matrix))
"""