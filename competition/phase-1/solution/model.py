# This is a sample code submission.
# It is a simple machine learning classifier.

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class Model:
    def __init__(self, metadata):
        """<ADD DOCUMENTATION HERE>"""
        self.classifier = DecisionTreeClassifier()

    def train(self, X, y):
        """Train the model.

        Args:
            X: Training data matrix of shape (num-samples, num-features), type np.ndarray.
            y: Training label vector of shape (num-samples), type np.ndarray.
        """
        print("FROM MODEL.PY:")
        print(f"X has shape {X.shape} and is:\n{X}")
        print(f"y has shape {y.shape} and is:\n{y}")
        self.classifier.fit(X, y)

    def test(self, X):
        """Predict labels.

        Args:
          X: Data matrix of shape (num-samples, num-features) to pass to the model for inference, type np.ndarray.
        """
        y = self.classifier.predict(X)
        return y
