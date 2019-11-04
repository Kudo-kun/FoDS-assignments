"""
dataset : https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+(North+Jutland%2C+Denmark)
"""

import numpy as np
from math import sqrt


class RegressionModel:
    def __init__(self, N, X, Y, x, y, xval, yval):
        """
        X :: training data                  (304113 x 3)
        x :: testing data                   (43786 x 3)
        Y :: training target values         (304113 x 1)
        y :: testing target values          (43786 x 1)
        xval :: validation data             (86975 x 3)
        yval :: validation training data    (86975 X 1)
        """
        self.N = N
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.x = np.array(x)
        self.y = np.array(y)
        self.xval = np.array(xval)
        self.yval = np.array(yval)

    def score(self, weights):
        """
        the following method helps us find the
        R2 (R-squared) error of a given training data
        wrt the generated weights
        """
        ss_tot = sum(np.square(np.mean(self.y) - self.y))
        ss_res = sum(np.square(self.x @ weights) - self.y)
        rmse = sqrt(0.5 * sum(np.square(self.x @ weights - self.y)))
        r2 = (1-(ss_res / ss_tot))
        return [r2*100, rmse]

    def gradient_descent(self):
        """
        train till error is almost constant
        """
        lr = 8.5*(10 ** -7)
        prev_err, count = (10 ** 10), 0
        W = np.random.random(self.N)
        while True:
            diff = ((self.X @ W) - self.Y)
            err = 0.5 * (diff @ diff)
            grad = (self.X.T @ diff)
            if count % 250 == 0:
                print("epoch =", count, "\nerr_diff =", prev_err-err)
                print("error =", err, "\n", W)
                print("score =", self.score(W), end="\n\n")
            W -= lr * grad
            if abs(prev_err-err) <= 0.00001:
                break
            prev_err = err
            count += 1
        print(count, err)
        print(W, self.score(W), end="\n\n")
        
    def stocastic_gradient_descent(self, epochs):
        """
        train till error is almost constant
        """
        lr = 5*(10 ** -3)
        W = np.random.randn(self.N)
        for count in range(epochs):
            diff = ((self.X @ W) - self.Y)
            err = 0.5 * (diff @ diff)
            for j in range(self.N):
                W[j] -= lr * (self.X[count].dot(W.T) -
                              self.Y[count]) * self.X[count][j]
            if count % 250 == 0:
                print("epoch =", count)
                print("error =", err, "\n", W)
                print("score =", self.score(W), end="\n\n")

    def gradient_descent_L1_reg(self):
        """
        attempts a L1 regularization on the data
        considering 10% of training data as validation data
        """
        W_fin = []
        lr, l1_fin = 5*(10 ** -7), 0
        MVLE = (10 ** 10)
        L1_vals = [0.0, 0.05, 0.15, 0.25, 0.35, 0.45,
                   0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
        sgn = lambda x: (x / abs(x))
        for l1 in L1_vals:
            prev_err, count = (10 ** 10), 0
            W = np.random.randn(self.N)
            while True:
                diff = ((self.X @ W) - self.Y)
                err = 0.5 * ((diff @ diff) + l1*sum([abs(w) for w in W]))
                if count % 250 == 0:
                    print("L1 hyperparamter =", l1, end=", ")
                    print("epoch =", count, "\nerr_diff =", prev_err-err)
                    print("error = ", err, "\n", W)
                    print("score =", self.score(W), end="\n\n")
                sgn_w = np.array([sgn(w) for w in W])
                W -= lr * ((self.X.T @ diff) + 0.5*l1*sgn_w)
                if abs(prev_err-err) <= 0.05:
                    break
                prev_err = err
                count += 1
            VLD = ((self.xval @ W) - self.yval)
            VLE = 0.5 * ((VLD.T @ VLD) + l1*sum([abs(w) for w in W]))
            if VLE < MVLE:
                W_fin = W
                l1_fin = l1
                MVLE = VLE
        print(MVLE, l1_fin, W_fin)

    def gradient_descent_L2_reg(self):
        """
        attempts a L2 regularization on the data
        considering 10% of training data as validation data
        """
        W_fin = []
        lr, l2_fin = 5*(10 ** -7), 0
        MVLE = (10 ** 10)
        L2_vals = [0.05, 0.15, 0.25, 0.35, 0.45,
                   0.55, 0.65, 0.75, 0.85, 0.95]
        for l2 in L2_vals:
            prev_err, count = (10 ** 10), 0
            W = np.random.randn(self.N)
            while True:
                diff = ((self.X @ W) - self.Y)
                err = 0.5 * ((diff @ diff) + l2*sum([w*w for w in W]))
                if count % 250 == 0:
                    print("L2 hyperparamter =", l2, end=", ")
                    print("epoch =", count, "\nerr_diff =", prev_err-err)
                    print("error = ", err, "\n", W)
                    print("score =", self.score(W), end="\n\n")
                W -= lr * ((self.X.T @ diff) + l2*W)
                if abs(prev_err-err) <= 0.05:
                    break
                prev_err = err
                count += 1
            VLD = ((self.xval @ W) - self.yval)
            VLE = 0.5 * ((VLD.T @ VLD) + l2 * (W.T @ W))
            if VLE < MVLE:
                W_fin = W
                l2_fin = l2
                MVLE = VLE
        print(MVLE, l2_fin, W_fin)

    def fit(self):
        """
        solves for optimal weights using system of
        N linear equations; AW = B, hence, W = inv(A)*B
        """
        B = self.X.T @ self.Y
        A = self.X.T @ self.X
        W = (np.linalg.inv(A)) @ B
        print(W, self.score(W))
