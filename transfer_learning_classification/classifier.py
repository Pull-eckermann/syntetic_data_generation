import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

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