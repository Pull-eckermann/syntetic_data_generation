import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import numpy as np
import tensorflow as tf

def plot_trainig_grapth(acc, val_acc, loss, val_loss):
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

#================================================ Vars declaration ================================================

BATCH_SIZE = 32
N_EPOCHS = 15
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)
LEARNING_RATE = 0.0001
TRAINABLE = False
MINIMALISTIC = True

#================================================ Dataset treatment ================================================
parser = argparse.ArgumentParser(description='Transfer learning options')

parser.add_argument('dataset', metavar='<PKlot|CNR|all-synthetic>', default='PKlot', help='Dataset to be used in the fine tunning')
parser.add_argument('-m', '--mixed_sintetic_data', default=False, action='store_true', help='Defines if the training will be made with sintetic data injection')
parser.add_argument('baseModel', metavar='<v3|v2>', default='v3', help='Base model to be used')
args = parser.parse_args()

data_op = args.dataset

if data_op == 'PKlot':
  if args.mixed_sintetic_data:
    train_dir = '../../Datasets/PKLot/All-data-mixed/Train'
    export_path = "retrained/saved_models/PKLot-mixed-"
    thresh_path = 'retrained/eer_threshods/PKLot-mixed-'
  else:
    train_dir = '../../Datasets/PKLot/All-data/Train'
    export_path = "retrained/saved_models/PKLot-"
    thresh_path = "retrained/eer_threshods/PKLot-"
  validation_dir = '../../Datasets/PKLot/All-data/Validation'

elif data_op == 'CNR':
  if args.mixed_sintetic_data:
    train_dir = '../../Datasets/CNRPark-EXT/All-data-mixed/Train'
    export_path = "retrained/saved_models/CNR-mixed-"
    thresh_path = "retrained/eer_threshods/CNR-mixed-"
  else:
    train_dir = '../../Datasets/CNRPark-EXT/All-data/Train'
    export_path = "retrained/saved_models/CNR-"
    thresh_path = "retrained/eer_threshods/CNR-"
  validation_dir = '../../Datasets/CNRPark-EXT/All-data/Validation'

elif data_op == 'all-synthetic':
  train_dir = '../../Datasets-synthetic/all-synthetic'
  export_path = "retrained/saved_models/all-synthetic-"
  thresh_path = "retrained/eer_threshods/all-synthetic-"
  validation_dir = '../../Datasets-synthetic/all-synthetic'

else:
  print("Please select one valid option. Exiting program...")
  exit(0)

# Read images from train directory
train_dataset = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  shuffle=True,
  #validation_split=0.2,
  #subset="training",
  #seed=123,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE
)

# Read images from validation directory
validation_dataset = tf.keras.utils.image_dataset_from_directory(
  validation_dir,
  shuffle=False,
  #validation_split=0.2,
  #subset="validation",
  #seed=123,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE
)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

#================================================ Base model expecification ================================================

model_op = args.baseModel

#Data augmentation for dataset
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.RandomContrast(0.4),
])

if model_op == 'v3':
  export_path = export_path + 'v3'
  thresh_path = thresh_path + 'v3.txt'

  # Create the base model from the pre-trained model MobileNet V3
  base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=IMG_SHAPE,
    include_top=False, # Remove the classification layer
    weights='imagenet',
    dropout_rate=0.2,
    minimalistic=MINIMALISTIC,
  )

  # Freeze the layers to not change the weigths
  base_model.trainable = TRAINABLE
  
  # Let's take a look at the base model architecture
  #base_model.summary()

  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  prediction_layer = tf.keras.layers.Dense(1)

  inputs = tf.keras.Input(shape=IMG_SHAPE)
  x = data_augmentation(inputs)
  x = base_model(x, training=False)
  x = global_average_layer(x)
  #x = tf.keras.layers.Dropout(0.2)(x)
  outputs = prediction_layer(x)
  model = tf.keras.Model(inputs, outputs)
  
elif model_op == 'v2':
  export_path = export_path + 'v2'
  thresh_path = thresh_path + 'v2.txt'

  # Create the base model from the pre-trained model MobileNet V3
  preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
  base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False, # Remove the classification layer
    weights='imagenet',
  )

  # Freeze the layers to not change the weigths
  base_model.trainable = False

  # Let's take a look at the base model architecture
  #base_model.summary()

  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  prediction_layer = tf.keras.layers.Dense(1)

  inputs = tf.keras.Input(shape=IMG_SHAPE)
  x = data_augmentation(inputs)
  x = preprocess_input(x)
  x = base_model(x, training=False)
  x = global_average_layer(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  outputs = prediction_layer(x)
  model = tf.keras.Model(inputs, outputs)

else:
  print("Please select one of the two models. exiting program...")
  exit(0)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

#================================================ Start training ================================================
earlyStopping= tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=3,
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
  
#================================================ Generate training graph ================================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

#plot_trainig_grapth (acc, val_acc, loss, val_loss)

#Saves the model
model.save(export_path)

labels = np.concatenate([y for x, y in validation_dataset], axis=0)
predictions = model.predict(validation_dataset).ravel()
predictions = tf.nn.sigmoid(predictions)

fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)

eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
threshold = interp1d(fpr, thresholds)(eer)

print("Equal Error Rate threshold: ", threshold)

with open(thresh_path, 'w') as file_out:
    file_out.write(threshold)
