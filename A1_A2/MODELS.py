import pandas as pd
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
        ss_res = sum(np.square((self.x @ weights) - self.y))
        test_err = (0.5/len(self.x)) * ss_res
        print("avg. test err =", test_err)
        rmse = sqrt(ss_res/len(self.x))
        r2 = (1-(ss_res/ss_tot))
        return [r2*100, rmse]

    def gradient_descent(self):
        """
        train till error is almost constant
        """
        prev_err, lr = 1e10, 3e-6
        W = np.ones(self.N) * 10
        for _ in range(50001):
            diff = ((self.X @ W) - self.Y)
            err = 0.5 * np.sum(np.square(diff))
            grad = (self.X.T @ diff)
            if _ % 500 == 0:
                print("epoch =", _, "| err_diff =", prev_err-err)
                print("avg. train err = ", err/(len(self.X)), "||", W)
                print("score =", self.score(W), end="\n\n")
            W -= lr * grad
            # if abs(prev_err-err) <= 5e-8:
            #     break
            prev_err = err
        print(err)
        print(W, self.score(W), end="\n\n")
        
    def stocastic_gradient_descent(self, epochs):
        """
        train till error is almost constant
        """
        lr = 0.05
        W = self.fit()
        for _ in range(epochs):
            diff = ((self.X @ W) - self.Y)
            err = 0.5 * (diff @ diff)
            idx = np.random.randint(0, len(self.X))
            W -= lr * (((self.X[idx] @ W) - self.Y[idx]) * self.X[idx])
            if _ % 500 == 0:
                print("epoch =", _)
                print("error =", err, "||", W)
                print("score =", self.score(W), end="\n\n")
        
    def gradient_descent_L1_reg(self):
        """
        attempts a L1 regularization on the data
        considering 10% of training data as validation data
        """
        W_fin = np.array([])
        lr, l1_fin = 1e-6, 0
        MVLE = 1e10
        L1_vals = np.linspace(0.0, 1.0, 9)
        sgn = lambda x: (x / abs(x))
        for l1 in L1_vals:
            prev_err = 1e10
            W = np.random.randn(self.N)
            for _ in range(50001):
                diff = ((self.X @ W) - self.Y)
                err = 0.5 * ((diff @ diff) + l1*sum([abs(w) for w in W]))
                if _ % 500 == 0:
                    print("L1 hyperparamter =", l1, end=", ")
                    print("epoch =", _, "| err_diff =", prev_err-err)
                    print("error = ", err, "||", W)
                    print("score =", self.score(W), end="\n\n")
                sgn_w = np.array([sgn(w) for w in W])
                W -= lr * ((self.X.T @ diff) + 0.5*l1*sgn_w)
                if abs(prev_err-err) < 0.01:
                    break
                prev_err = err
            VLD = ((self.xval @ W) - self.yval)
            VLE = 0.5 * ((VLD @ VLD) + l1*sum([abs(w) for w in W]))
            if VLE < MVLE:
                W_fin = W
                l1_fin = l1
                MVLE = VLE
        print(MVLE, l1_fin, W_fin)
        print(self.score(W_fin))

    def gradient_descent_L2_reg(self):
        """
        attempts a L2 regularization on the data
        considering 10% of training data as validation data
        """
        W_fin = np.array([])
        lr, l2_fin = 1e-6, 0
        MVLE = 1e10
        L2_vals = np.linspace(0.0, 1.0, 9)
        for l2 in L2_vals:
            prev_err, count = 1e10, 0
            W = np.ones(self.N)*10
            for _ in range(50001):
                diff = ((self.X @ W) - self.Y)
                err = 0.5 * ((diff @ diff) + l2*sum([w*w for w in W]))
                if _ % 500 == 0:
                    print("L2 hyperparamter =", l2, end=", ")
                    print("epoch =", _, "| err_diff =", prev_err-err)
                    print("error = ", err, "||", W)
                    print("score =", self.score(W), end="\n\n")
                W -= lr * ((self.X.T @ diff) + l2*W)
                if abs(prev_err-err) < 0.01:
                    break
                prev_err = err
            VLD = ((self.xval @ W) - self.yval)
            VLE = 0.5 * ((VLD @ VLD) + l2*(W @ W))
            if VLE < MVLE:
                W_fin = W
                l2_fin = l2
                MVLE = VLE
        print(MVLE, l2_fin, W_fin)
        print(self.score(W_fin))

    def fit(self, lam):
        """
        solves for optimal weights using system of
        N linear equations; AW = B, hence, W = inv(A)*B
        """
        B = self.X.T @ self.Y
        A = ((self.X.T @ self.X) + (lam * np.identity(self.N)))
        W = (np.linalg.inv(A)) @ B
        print(W, self.score(W))
        return W
