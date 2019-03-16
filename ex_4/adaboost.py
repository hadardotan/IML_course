"""
hadar.dotan


===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Noga Zaslavsky
Edited: Yoav Wald, May 2018

"""
import numpy as np

class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        m,d = X.shape
        Dt = np.array([1/m]*m)
        for t in range(self.T):
            # updates h - by the end of the loop we get ht according to
            #  adaboost
            self.h[t] = self.WL(Dt,X,y)
            Ld = []
            for i in range(len(y)):
                if self.h[t].predict(X)[i] != y[i]:
                    Ld.append(Dt[i])
            et = np.sum(Ld)
            self.w[t] = 0.5 * np.log((1/et) - 1)
            Dt = [Dt[i]*np.exp(-1*self.w[t]*y[i]*self.h[t].predict(X)[i]) for i in range(len(y))]
            Dt /= np.sum(Dt)


    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        # hs = np.zeros(self.T)
        hs = [self.h[t].predict(X) for t in range(self.T)]
        hs = np.column_stack(hs)
        y_hat = np.array(np.sign(np.dot(hs,self.w)))
        return y_hat

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """

        y_hat = self.predict(X)
        error = np.sum(1 for i in range(len(y)) if y_hat[i] != y[i])
        mean_error = error / len(y)
        return mean_error

