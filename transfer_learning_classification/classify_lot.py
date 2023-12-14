import numpy as np
import tensorflow as tf
import sys, os
from sklearn import metrics
import PIL.Image as Image

#===============================
if len(sys.argv) < 2:
    print("SINTAX ERROR: python3 classify_lot.py <directory path>")
    exit(0)

#List the names of classes encountered
class_names = ['Empty', 'Occupied']
class_names = np.array(class_names)

# Restore the model
export_path = "retrained/saved_models/UFPR04-real"
model = tf.keras.models.load_model(export_path)

# Prepare Test set
IMAGE_SHAPE = (160, 160)

predicted_labels = []
labels = []

lots = os.walk(sys.argv[1])
for path, _, files in lots:
    for img in files:
        pk_space = path + "/" + img #Image that will be classified
        pk_space = Image.open(pk_space).resize(IMAGE_SHAPE)

        # Normalization
        pk_space = np.array(pk_space)/255.0

        # Add a batch dimension (with np.newaxis) and pass the image to the model:
        result = model.predict(pk_space[np.newaxis, ...])

        #The result is a 1001-element vector of logits, rating the probability of each class for the image.
        #The top class ID can be found with tf.math.argmax:
        predicted_class = tf.math.argmax(result[0], axis=-1)

        ###Decode the prediction and store in a list
        predicted_labels.append(class_names[predicted_class])
        if 'Empty' in path:
            labels.append('Empty')
        elif 'Occupied' in path:
            labels.append('Occupied')

acc = metrics.accuracy_score(labels, predicted_labels)
print('Test accuracy: {}'.format(acc))