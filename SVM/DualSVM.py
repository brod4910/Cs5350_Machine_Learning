import scipy.optimize as optimize
import numpy as np


class DualSVM():
    def __init__(self, C, lr= 1, kernel= None):
        self.C = C
        self.lr = lr
        self.alpha = None
        self.kernel= kernel

    def train_dataset(self, X, Y):
        if self.alpha is None:
            self.alpha = np.full((X.shape[0]), self.C)

        if self.kernel == 'gaussian':
            gram_m = self._gram_matrix(X)
            args = (X, Y, gram_m)
        else:
            xx = x.T.dot(x)
            args = (X, Y, xx)

        cons = [{'type': 'eq', 'fun' : self.constraint_1, 'args' : args}]

        b = ((0, self.C),)
        bounds = ()
        for i in range(X.shape[0]):
            bounds = bounds + b     
        
        res = optimize.minimize(self._quadratic_convex_fn, self.alpha, args= args, bounds= bounds, constraints= cons, method= 'SLSQP')

        self.alpha = res.x

        ay = self.alpha*Y

        ay = ay.reshape((1, ay.shape[0]))

        self.weights = ay.dot(X)
        print(self.weights)

    def _quadratic_convex_fn(self, a, *args):
        x = args[0]
        y = args[1]

        xx = args[2]

        yy = np.multiply(y,y)
        aa = np.multiply(a, a)

        yax = np.sum(np.multiply(yy, aa) * xx)

        objective = (yax - np.sum(a))/2
        return objective

    def constraint_1(self, a, *args):
        return np.sum(np.multiply(a, args[1]))

    def test_dataset(self, X, Y):
        corr = 0
        predictions = []
        for (x, y) in zip(X, Y):
            x_p, y_p = x.reshape(x.shape[0], 1), y.reshape(y.shape[0], 1)
            y_prime = self._test_predict(x)
            predictions.append(y_prime)

            if y_prime == y:
                corr += 1
        err = 1 - corr/len(Y)

        return predictions, err

    def _test_predict(self, x):
        y_prime = np.sign(np.dot(self.weights, x))
        return y_prime

    def _gaussian(self, x, y, lr= .01):
        exponent = -np.sqrt(np.linalg.norm(x-y) ** 2 /lr)
        return np.exp(exponent)

    def _gram_matrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._gaussian(x_i, x_j)
        return K

