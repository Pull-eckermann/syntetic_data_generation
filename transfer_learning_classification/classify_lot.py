import numpy as np
import tensorflow as tf
import sys, os
from sklearn import metrics


#===============================
if len(sys.argv) < 2:
    print("SINTAX ERROR: python3 classify_lot.py <directory path>")
    exit(0)

#List the names of classes encountered
class_names = ['Empty', 'Occupied']
class_names = np.array(class_names)

# Restore the model
export_path = "retrained/saved_models/parking_lot"
model = tf.keras.models.load_model(export_path)

# Prepare Test set
IMAGE_SHAPE = (224, 224)
img_height = 224
img_width = 224
batch_size = 16

test_ds = tf.keras.utils.image_dataset_from_directory(
  str(sys.argv[1]),
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# Normalize data
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.

scores = []
for image_batch, true_id in test_ds:
    predicted_batch = model.predict(image_batch)
    predicted_id = tf.math.argmax(predicted_batch, axis=-1)
    predicted_labels = class_names[predicted_id]
    labels = class_names[true_id]
    scores.append(metrics.accuracy_score(labels, predicted_labels))

print('Total accuracy: {}'.format(sum(scores) / len(scores)))

#predicted_labels = []
#labels = []
#
#lots = os.walk(sys.argv[1])
#for path, _, files in lots:
#    for img in files:
#        pk_space = path + "/" + img #Image that will be classified
#        pk_space = Image.open(pk_space).resize(IMAGE_SHAPE)
#
#        # Normalization
#        pk_space = np.array(pk_space)/255.0
#
#        # Add a batch dimension (with np.newaxis) and pass the image to the model:
#        result = model.predict(pk_space[np.newaxis, ...], verbose = 0)
#
#        #The result is a 1001-element vector of logits, rating the probability of each class for the image.
#        #The top class ID can be found with tf.math.argmax:
#        predicted_class = tf.math.argmax(result[0], axis=-1)
#
#        ###Decode the prediction and store in a list
#        predicted_labels.append(class_names[predicted_class])
#        if 'Empty' in path:
#            labels.append('Empty')
#        elif 'Occupied' in path:
#            labels.append('Occupied')
#
#acc = metrics.accuracy_score(labels, predicted_labels)
#print('Test accuracy: {}'.format(acc))