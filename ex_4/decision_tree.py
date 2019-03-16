"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the decision tree classifier with real-values features.
Training algorithm: CART

Author: Noga Zaslavsky
Edited: Yoav Wald, May 2018

"""
import numpy as np

class Node(object):
    """ A node in a real-valued decision tree.
        Set all the attributes properly for viewing the tree after training.
    """
    def __init__(self,leaf = True,left = None,right = None,samples = 0,feature = None,theta = 0.5,misclassification = 0,label = None):
        """
        Parameters
        ----------
        leaf : True if the node is a leaf, False otherwise
        left : left child
        right : right child
        samples : number of training samples that got to this node
        feature : a coordinate j in [d], where d is the dimension of x (only for internal nodes)
        theta : threshold over self.feature (only for internal nodes)
        label : the label of the node, if it is a leaf
        """
        self.leaf = leaf
        self.left = left
        self.right = right
        self.samples = samples
        self.feature = feature
        self.theta = theta
        self.misclassification = misclassification
        self.label = label


class DecisionTree(object):
    """ A decision tree for binary classification.
        max_depth - the maximum depth allowed for a node in this tree.
        Training method: CART
    """

    def __init__(self,max_depth):
        self.root = None
        self.max_depth = max_depth

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        self.root = self.CART(X, y, X.T, 0)

    def left_rect(self, X, d, s):
        return X[:, d] <= s

    def right_rect(self, X, d, s):
        return X[:, d] > s

    def CART(self,X, y, A, depth):
        """
        Gorw a decision tree with the CART method ()

        Parameters
        ----------
        X, y : sample
        A : array of d*m real features, A[j,:] row corresponds to thresholds over x_j
        depth : current depth of the tree

        Returns
        -------
        node : an instance of the class Node (can be either a root of a subtree or a leaf)
        """

        #init root
        root_node = Node()
        root_node.samples = X.shape[0]

        if depth == self.max_depth:

            bin = np.bincount(y.astype(np.int) + 1)

            if bin.size == 0:
                root_node.label = -1
            else:
                root_node.label = np.argmax(bin) - 1
            return root_node

        min_err, m = X.shape[0], X.shape[0]
        min_j = 0
        min_s = 0
        best_r_r = [True]*X.shape[0]
        best_l_r = [True]*X.shape[0]

        for d, j in enumerate(X.T):
            for s in j:
                r_rect = self.right_rect(X, d, s)
                l_rect = self.left_rect(X, d, s)

                if y[r_rect].shape[0] != m and y[r_rect].shape[0] != 0:
                    label_r = np.argmax(np.bincount(y[r_rect].astype(np.int) + 1)) - 1
                    label_l = np.argmax(np.bincount(y[l_rect].astype(np.int) + 1)) - 1
                    curr_err = np.sum(y[r_rect] != label_r) + np.sum(y[l_rect] != label_l)

                else:
                    label = np.argmax(np.bincount(y.astype(np.int) + 1)) - 1
                    curr_err = np.sum(y != label)

                if curr_err < min_err:
                    min_err = curr_err
                    min_j = d
                    min_s = s
                    best_r_r = r_rect
                    best_l_r = l_rect

        if min_err == 0 or np.count_nonzero(best_r_r) == X.shape[0] or X[best_l_r].shape[0] == 0:
            bin = np.bincount(y.astype(np.int) + 1)
            if bin.size == 0:
                root_node.label = -1
            else:
                root_node.label = np.argmax(bin) - 1
            return root_node

        root_node.feature = min_j
        root_node.theta = min_s
        root_node.misclassification = min_err

        left_node = self.CART(X[best_r_r], y[best_r_r], A, depth + 1)
        right_node = self.CART(X[best_l_r], y[best_l_r], A, depth + 1)

        root_node.leaf = False
        root_node.left = left_node
        root_node.right = right_node

        return root_node

    def node_prediction(self, node, X):
        if node.leaf:
            return [node.label]*X.shape[0]

        d = node.feature
        s = node.theta
        y = np.zeros(X.shape[0])
        r_rect = self.right_rect(X, d, s)
        l_rect = self.left_rect(X, d, s)

        y[r_rect] = self.node_prediction(node.left, X[r_rect])
        y[l_rect] = self.node_prediction(node.right, X[l_rect])

        return y

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        return np.array(self.node_prediction(self.root, X))


    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        return (1/X.shape[0]) * np.count_nonzero(y - self.predict(X))
