import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from keras.callbacks import Callback


train_dir = '../../Datasets/PKLot-rotated-rect/All-data/Train'
validation_dir = '../../Datasets/PKLot-rotated-rect/All-data/Test'


BATCH_SIZE = 32
IMG_SIZE = (160, 160)

"""
train_dataset = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  shuffle=True,
  validation_split=0.3,
  subset="training",
  seed=123,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
  validation_dir,
  validation_split=0.3,
  subset="validation",
  seed=123,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE
)
"""

# Read images from train directory
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

# Read images from validation directory
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=False,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

# Separe a set of validation dataset for test
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#Data augmentation for dataset
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

# Pixels rescaling to mobilenetv2 
preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

# Create the base model from the pre-trained model MobileNet V3
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                               include_top=False, # Remove the classification layer
                                               weights='imagenet',
                                               include_preprocessing=True)
# Freeze the layers to not change the weigths
base_model.trainable = False

# Let's take a look at the base model architecture
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


#================ Start training ====================
initial_epochs = 15

# Test whitout training
loss0, accuracy0 = model.evaluate(test_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# Start training
# Callbacks to save the best model
earlyStopping= tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
saveBestModel = tf.keras.callbacks.ModelCheckpoint(filepath="retrained/weights.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset,
                    callbacks=[earlyStopping, saveBestModel])

model.load_weights('retrained/weights.hdf5')

#================ Generats training graph ====================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
"""
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
"""
#Saves the model
export_path = "retrained/saved_models/PKLot-rrect-v3"
model.save(export_path)

# Test after training
lossF, accuracyF = model.evaluate(test_dataset, verbose='True')
print("Final loss: {:.2f}".format(lossF))
print("Final accuracy: {:.2f}".format(accuracyF))