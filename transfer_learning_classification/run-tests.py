import numpy as np
import os
import tensorflow as tf

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
base_path = "retrained/saved_models/"

#models = ['CNR-bbox-v3', 'CNR-rrect-v3', 'PKLot-bbox-v3', 'PKLot-rrect-v3']
models = ['CNR-rrect-mixed-v3']


for model_name in models:
    print("###############################################################################")
    print("Model {} is being loaded for tests".format(model_name))
    # Restore the model
    export_path = base_path + model_name
    model = tf.keras.models.load_model(export_path)

    if 'CNR' in model_name:
        base_validation = "../../Datasets/Tests/pklot/"
    else:
        base_validation = "../../Datasets/Tests/cnrpark/"
    
    tests = os.listdir(base_validation)

    for test in tests:
        validation_dir = base_validation + test
        validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(validation_dir,
                                                                         shuffle=True,
                                                                         batch_size=BATCH_SIZE,
                                                                         image_size=IMG_SIZE)

        AUTOTUNE = tf.data.AUTOTUNE
        validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

        #Predict
        print(">>>>>> Executing evaluation for " + test)
        accuracy = loss = 0.0
        for i in range(1,4):
            print("{} execution:".format(i))
            tmp_loss, tmp_accuracy = model.evaluate(validation_dataset)
            accuracy += tmp_accuracy
            loss += tmp_loss
        print("Loss: {}".format(loss/3.0))
        print("Accuracy: {}".format(accuracy/3.0))
        print("====================================================================================")
