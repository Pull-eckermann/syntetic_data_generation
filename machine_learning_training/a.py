import sys
import os
from KnnClassifier import KnnClassifier

training_dataset = "nomalized_train_features.csv"
knn = KnnClassifier(3, training_dataset)

try:
    instance = sys.argv[1]
except:
    print('ERROR: Please especify the path to Test Dataset')
    exit(0)



image = instance # Path to the Parking Lot image
result = knn.classify(image) # Classify image with given knn
if result == 0:
    print("Parking space is free")
else:
    print("Parking space is occupied")