from KnnClassifier import KnnClassifier
from sklearn.neighbors import KNeighborsClassifier
from localbinarypatterns import LocalBinaryPatterns
from sklearn import metrics
import cv2
import sys
import os
import csv

if len(sys.argv) != 4:
    print("ERROR: Input invalid. Correct syntax: python3 classify_lot.py <mode> <number of neighbors> <Test set path>")
    exit(0)

training_dataset = "normalized_syntetic-set-3.csv"

k = int(sys.argv[2])
if sys.argv[1] == 'scikit':
    knn = KNeighborsClassifier(k)
elif sys.argv[1] == 'default':
    knn = KnnClassifier(k)

descriptor = LocalBinaryPatterns(8,1)

X_train = []
y_train = []
with open(training_dataset, 'r+', newline='') as train:
    reader = csv.reader(train)
    for row in reader:
        label = int(row.pop())
        X_train.append(list(map(float, row)))
        y_train.append(label)
knn.fit(X_train, y_train)

try:
    instance = sys.argv[3]
except:
    print('ERROR: Please especify the path to Test Dataset')
    exit(0)

Lots = os.walk(instance)
X_test = []
y_test = []

for path, _, files in Lots:
    for img in files:
        image = path + '/' + img # Path to the Parking Lot image
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, hist_test = descriptor.describe(image)
        hist_test = hist_test / 6034 # min max normalization
        X_test.append(hist_test)
        if 'Empty' in path:
            y_test.append(0)
        elif 'Occupied' in path:
            y_test.append(1)

y_pred = knn.predict(X_test)
score = metrics.accuracy_score(y_test,y_pred)

print('Predictor accuracy: ', score)