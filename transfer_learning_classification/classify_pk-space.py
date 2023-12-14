import numpy as np
import PIL.Image as Image
import matplotlib.pylab as plt
import tensorflow as tf
import sys, os


#===============================
if len(sys.argv) < 2:
    print("SINTAX ERROR: python3 classify_pk-space.py <IMAGE_NAME>")
    exit(0)

#List the names of classes encountered
class_names = ['Empty', 'Occupied']
class_names = np.array(class_names)

IMAGE_SHAPE = (160, 160)

# Restore the model
export_path = "retrained/saved_models/UFPR04-real"
model = tf.keras.models.load_model(export_path)

# Image that will be classified
directory = os.getcwd() + "/" #Get current directory
pk_space = directory + "/" + sys.argv[1] #Image that will be classified
pk_space = Image.open(pk_space).resize(IMAGE_SHAPE)

# Maybe this is essential
pk_space = np.array(pk_space)

# Add a batch dimension (with np.newaxis) and pass the image to the model:
result = model.predict_on_batch(pk_space[np.newaxis, ...]).flatten()

#The result is a 1001-element vector of logits, rating the probability of each class for the image.
#The top class ID can be found with tf.math.argmax:
#predicted_class = tf.math.argmax(result[0], axis=-1)
predicted_ids = tf.nn.sigmoid(result[0])
predicted_ids = tf.where(predicted_ids < 0.5, 0, 1)

###Decode the prediction
predicted_label = class_names[predicted_ids]

plt.imshow(pk_space)
plt.axis('off')
_ = plt.title("Prediction: " + predicted_label)
plt.show()