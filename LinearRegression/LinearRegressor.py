import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.error_functions import least_mean_squares, derv_LMS
import numpy as np
from copy import deepcopy

class LinearRegressor():
    def __init__(self, lr, weights):
        self.lr = lr
        self.weights = weights
        self.weights_t = None

    def train(self, x_i, y_i):
        pred = np.dot(x_i, self.weights)

        lms = self._calc_error(pred, y_i)

        d_LMS = self._update_weights(pred, y_i, x_i)

        convergence = np.linalg.norm(np.stack((self.weights, self.weights_t)))

        return lms, convergence

    def _update_weights(self, pred, y_i, x_i):
        d_LMS = derv_LMS(pred, y_i, x_i)
        self.weights_t = deepcopy(self.weights)
        self.weights = self.weights - (self.lr * d_LMS).T
        return d_LMS

    def test_batch(self, test_examples, test_labels):
        preds = []
        for ex in test_examples:
            preds.append(self._test(ex))
        return preds

    def _calc_error(self, preds, y_i):
        return least_mean_squares(preds, y_i)

    def _test(self, x_i):
        return np.dot(x_i, self.weights)