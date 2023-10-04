import numpy as np
import cv2
import csv
from localbinarypatterns import LocalBinaryPatterns

class KnnClassifier:
    def __init__(self, neighbors : int, train_path : str):
        self.neighbors = neighbors # Number of neighbors to compare
        self.train_dataset = train_path # Path to the training dataset
        self.descriptor = LocalBinaryPatterns(8,1)

    # Return the class of a given instance 
    def classify(self, path_test : str):
        distances = []

        # Read image and calculates its histogram, normalyzing it with min_max
        image = cv2.imread(path_test)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, hist_test = self.descriptor.describe(image)
        hist_test = hist_test / 2745

        # Calculates distances from the given test instance with all train dataset instances
        with open(self.train_dataset, 'r+', newline='') as train:
            reader = csv.reader(train)
            for row in reader:
                label = int(row.pop())
                hist_train = list(map(float, row))
                dist = np.sum(np.square((hist_train - hist_test))) # Euclidean distance
                distances.append((label, dist))

        # Sort dintances list and based on the first k_neighbors predict the class
        distances.sort(key=lambda x:x[1])
        distances = distances[:self.neighbors]
        labels = [label[0] for label in distances]
        prediction = max(labels,key=labels.count)

        return prediction



