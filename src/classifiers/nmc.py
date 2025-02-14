import numpy as np
from sklearn.metrics import pairwise_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids

    @property
    def class_labels(self):
        return self._class_labels

    def fit(self, xtr, ytr):
        labels = np.unique(ytr)
        self._centroids = np.zeros(shape=(labels.size, xtr.shape[1]))

        for i, label in enumerate(labels):
            self._centroids[i, :] = xtr[ytr == label, :].mean(axis=0)

        return self

    def predict(self, xts):
        if self._centroids is None:
            raise ValueError("Centroids not set. Run fit(x,y) first!")

        dist = pairwise_distances(xts, self._centroids)
        ypred = np.argmin(dist, axis=1)
        return ypred
