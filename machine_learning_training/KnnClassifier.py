import numpy as np

class KnnClassifier:
    def __init__(self, neighbors : int):
        self.neighbors = neighbors # Number of neighbors to compare
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


    # Return the prediction of a given list of instances
    def predict(self, X_test):
        # List of predictios
        y_pred = []
        for instance in X_test:
            distances = []
            # Calculates distances from the given test instance with all train dataset instances
            for i in range(0, len(self.X_train)):
                label = self.y_train[i]
                train_data = self.X_train[i]
                dist = np.sum(np.square((train_data - instance))) # Euclidean distance
                distances.append((label, dist))

            # Sort dintances list and based on the first k_neighbors predict the class
            distances.sort(key=lambda x:x[1])
            distances = distances[:self.neighbors]
            labels = [label[0] for label in distances]
            prediction = max(labels,key=labels.count)
            y_pred.append(prediction)

        return y_pred



