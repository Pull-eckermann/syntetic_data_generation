from sklearn.metrics import accuracy_score
import sys
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
base_path = "retrained/saved_models/"
thresh_path = "retrained/eer_threshods/"

#model_name = 'PKLot-v3'
model_name = 'CNR-v3'
#model_name = 'PKLot-mixed-v3'
#model_name = 'CNR-mixed-v3'

print("###############################################################################")
print("Model {} is being loaded for tests".format(model_name))
# Restore the model
export_path = base_path + model_name
thresh_path = thresh_path + model_name + '.txt'

model = tf.keras.models.load_model(export_path)
with open(thresh_path, 'r') as thresh_file:
    threshold = float(thresh_file.read())
    
validation_dir = sys.argv[1]
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(validation_dir,
                                                                 shuffle=False,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)
AUTOTUNE = tf.data.AUTOTUNE
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

#Predict
print(">>>>>> Executing evaluation" )
labels = np.concatenate([y for x, y in validation_dataset], axis=0)
predictions = model.predict(validation_dataset).ravel()
predictions = tf.nn.sigmoid(predictions)

predicted_labels = tf.where(predictions < threshold, 0, 1)
predicted_labels = list(predicted_labels)
print("Accuracy: ", accuracy_score(labels, predicted_labels))
print("====================================================================================")
