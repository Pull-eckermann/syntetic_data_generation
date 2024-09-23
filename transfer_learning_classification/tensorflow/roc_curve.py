from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

BATCH_SIZE = 32
IMG_SIZE = (128, 128)

export_path = "retrained/saved_models/CNR-v3"

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

labels = np.concatenate([y for x, y in validation_dataset], axis=0)
predictions = model.predict(validation_dataset).ravel()
predictions = tf.nn.sigmoid(predictions)

fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
auc = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig("cam1-roc.jpeg")
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.savefig("cam1-roc-zoom.jpeg")

fnr = 1 - tpr
#EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
#EER2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
#eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
#threshold = interp1d(fpr, thresholds)(eer)

print("Equal Error Rate: ", threshold)

predicted_labels = tf.where(predictions < threshold , 0, 1)
predicted_labels = list(predicted_labels)
print("Total accuracy: ", accuracy_score(labels, predicted_labels))
