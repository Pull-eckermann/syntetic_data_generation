from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import numpy as np
import tensorflow as tf

#================================================ Vars declaration ================================================

BATCH_SIZE = 32
N_EPOCHS = 15
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)
LEARNING_RATE = 0.0001
TRAIN_DIR = '../../Datasets/tmp/cnr/Train'
VALIDATION_DIR = '../../Datasets/tmp/cnr/Validation'
EXPORT_PATH = 'retrained/saved_models/CNR-v3'
THRESH_PATH = 'retrained/eer_threshods/CNR-v3.txt'
TRAINABLE = True
MINIMALISTIC = False

#================================================ Input treatment ================================================

# Read images from train directory
train_dataset = tf.keras.utils.image_dataset_from_directory(
  TRAIN_DIR,
  shuffle=True,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE
)

# Read images from validation directory
validation_dataset = tf.keras.utils.image_dataset_from_directory(
  VALIDATION_DIR,
  shuffle=False,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE
)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

#================================================ Base model expecification ================================================

#Data augmentation for dataset
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.RandomContrast(0.4),
])

# Create the base model from the pre-trained model MobileNet V3
base_model = tf.keras.applications.MobileNetV3Large(
  input_shape=IMG_SHAPE,
  include_top=False, # Remove the classification layer
  weights='imagenet',
  minimalistic=MINIMALISTIC,
  pooling='avg',
  dropout_rate=0.2
)

# Freeze the layers to not change the weigths
base_model.trainable = TRAINABLE

#global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1, bias_initializer=tf.keras.initializers.VarianceScaling(
                                                              scale=0.3334,
                                                              mode='fan_in',
                                                              distribution='uniform'))

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = base_model(x, training=False)
#x = global_average_layer(x)
#x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

#================================================ Start training ================================================
earlyStopping= tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=5,
                                                verbose=1,
                                                mode='auto')

saveBestModel = tf.keras.callbacks.ModelCheckpoint(filepath="retrained/weights.hdf5",
                                                   monitor='val_loss',
                                                   verbose=1,
                                                   save_best_only=True,
                                                   mode='auto')

history = model.fit(train_dataset,
                    epochs=N_EPOCHS,
                    validation_data=validation_dataset,
                    callbacks=[earlyStopping, saveBestModel])

model.load_weights('retrained/weights.hdf5')
model.save(EXPORT_PATH)
  
labels = np.concatenate([y for x, y in validation_dataset], axis=0)
predictions = model.predict(validation_dataset).ravel()
predictions = tf.nn.sigmoid(predictions)

fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)

fnr = 1 - tpr
threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]

print("Equal Error Rate threshold: ", threshold)

with open(THRESH_PATH, 'w') as file_out:
    file_out.write(str(threshold))