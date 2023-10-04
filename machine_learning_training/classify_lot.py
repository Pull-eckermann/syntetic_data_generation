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

Lots = os.walk(instance)

sucess = 0
fails = 0
cont = 0 

for path, _, files in Lots:
    if cont >= 1000:
        break
    for img in files:
        image = path + '/' + img # Path to the Parking Lot image
        result = knn.classify(image) # Classify image with given knn
        if result == 0:
            if 'Empty' in path:
                sucess += 1
            else:
                fails += 1
        else:
            if 'Occupied' in path:
                sucess += 1
            else:
                fails += 1
        cont += 1
        if cont >= 1000:
            break


total = sucess + fails
acc = sucess / total

print('Total images: ', total)
print('Predictor accuracy: ', acc)