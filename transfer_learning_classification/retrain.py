import matplotlib.pyplot as plt
import os
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
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)
LEARNING_RATE = 0.0001

#================================================ Dataset treatment ================================================
os.system('clear')
print('>>> Which Dataset will be used in the training?')
print('1- PKLot-rotated-rects')
print('2- PKLot-bounding-boxes')
print('3- CNRPark-rotated-rects')
print('4- CNRPark-rotated-rects-mixed')
print('5- CNRPark-bounding-boxes')
data_op = input()

if data_op == '1':
  train_dir = '../../Datasets/PKLot-rotated-rects/All-data/Train'
  validation_dir = '../../Datasets/PKLot-rotated-rects/All-data/Test'
  export_path = "retrained/saved_models/PKLot-rrect-"
elif data_op == '2':
  train_dir = '../../Datasets/PKLot-bounding-boxes/All-data/Train'
  validation_dir = '../../Datasets/PKLot-bounding-boxes/All-data/Test'
  export_path = "retrained/saved_models/PKLot-bbox-"
elif data_op == '3':
  train_dir = '../../Datasets/CNRPark-EXT-rotated-rects/All-data/Train'
  validation_dir = '../../Datasets/CNRPark-EXT-rotated-rects/All-data/Test'
  export_path = "retrained/saved_models/CNR-rrect-"
elif data_op == '4':
  train_dir = '../../Datasets/CNRPark-EXT-rotated-rects/All-data-mixed/Train'
  validation_dir = '../../Datasets/CNRPark-EXT-rotated-rects/All-data/Test'
  export_path = "retrained/saved_models/CNR-rrect-mixed-"
elif data_op == '5':
  train_dir = '../../Datasets/CNRPark-EXT-bounding-boxes/All-data/Train'
  validation_dir = '../../Datasets/CNRPark-EXT-bounding-boxes/All-data/Test'
  export_path = "retrained/saved_models/CNR-bbox-"
else:
  print("Please select one valid option. Exiting program...")
  exit(0)

# Read images from train directory
train_dataset = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  shuffle=True,
  #validation_split=0.3,
  #subset="training",
  #seed=123,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE
)

# Read images from validation directory
validation_dataset = tf.keras.utils.image_dataset_from_directory(
  validation_dir,
  shuffle=False,
  #validation_split=0.3,
  #subset="validation",
  #seed=123,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE
)

# Separe a set of validation dataset for test
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#================================================ Base model expecification ================================================

os.system('clear')
print('>>> Select the base model to be retrained')
print('1- MobileNetV3')
print('2- MobileNetV2')
model_op = input()

#Data augmentation for dataset
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.RandomContrast(0.4),
])

if model_op == '1':
  export_path = export_path + 'v3'

  # Create the base model from the pre-trained model MobileNet V3
  base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=IMG_SHAPE,
    include_top=False, # Remove the classification layer
    weights='imagenet',
    minimalistic=True,
    #pooling='avg',
    #dropout_rate=0.2
  )

  # Freeze the layers to not change the weigths
  base_model.trainable = False
  
  # Let's take a look at the base model architecture
  base_model.summary()

  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  prediction_layer = tf.keras.layers.Dense(1)

  inputs = tf.keras.Input(shape=IMG_SHAPE)
  x = data_augmentation(inputs)
  x = base_model(x, training=False)
  x = global_average_layer(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  outputs = prediction_layer(x)
  model = tf.keras.Model(inputs, outputs)
  
elif model_op == '2':
  export_path = export_path + 'v2'

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
  base_model.summary()

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
initial_epochs = 15

# Test whitout training
loss0, accuracy0 = model.evaluate(test_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# Start training
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
                    epochs=initial_epochs,
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

# Test after training
lossF, accuracyF = model.evaluate(test_dataset, verbose='True')
print("Final loss: {:.2f}".format(lossF))
print("Final accuracy: {:.2f}".format(accuracyF))
