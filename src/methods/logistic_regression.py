import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500, task_kind = 'classification'):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.task_kind = task_kind
        self.w = None

    def softmax(self, data, W):
        val = np.exp(data @ W)
        s = np.sum(val, axis=1)
        return (val.T / s).T

    def loss_logistic_multi(self, data, labels, w):
        softmax = self.softmax(data, w)
        return - np.sum(labels * np.log(softmax))

    def gradient_logistic_multi(self, data, labels, W):
        return data.T @ (self.softmax(data, W) - labels)

    def logistic_regression_predict_multi(self, data, W):
        return np.argmax(self.softmax(data, W), axis=1)

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        labels_onehot = label_to_onehot(training_labels)
        D = training_data.shape[1]  # number of features
        C = labels_onehot.shape[1]  # number of classes
        # Random initialization of the weights
        weights = np.random.normal(0, 0.1, (D, C))
        for it in range(self.max_iters):
            gradient = self.gradient_logistic_multi(training_data, labels_onehot, weights)
            weights = weights - self.lr * gradient
        self.w = weights
        return onehot_to_label(self.softmax(training_data, weights))
        
    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        return onehot_to_label(self.softmax(test_data, self.w))
