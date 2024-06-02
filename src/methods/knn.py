import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind
        self.training_data = None
        self.training_labels = None

    def euclidean_dist(self, example, training_examples):
      return np.sqrt(np.sum((example - training_examples)**2, axis=1))

    def find_nearest_neighbors(self, distances):
      indices = np.argsort(distances)[:self.k]
      return indices
    
    def predict_label(self, neighbor_labels):
      if self.task_kind == "regression":
          return np.mean(neighbor_labels, axis=0)
      else:
          return np.argmax(np.bincount(neighbor_labels))

    def kNN_one(self, unlabeled_sample, training_features, training_labels):
      distances = self.euclidean_dist(unlabeled_sample, training_features)
      nn_indices = self.find_nearest_neighbors(distances)
      neighbor_labels = training_labels[nn_indices]
      return self.predict_label(neighbor_labels)

    def kNN(self, unlabeled, training_features, training_labels):
      return np.apply_along_axis(func1d=self.kNN_one, axis=1, arr=unlabeled, 
                               training_features=training_features, 
                               training_labels=training_labels)

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        self.training_data = training_data
        self.training_labels = training_labels
        pred_labels = self.kNN(training_data, training_data, training_labels)
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        test_labels = self.kNN(test_data, self.training_data, self.training_labels)
        return test_labels