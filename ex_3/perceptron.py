import sklearn.svm as sv
import numpy as np
import sklearn.metrics as met
import  matplotlib.pyplot as plt


class Perceptron:
    """
    this class implements a Perceptron
    """

    def __init__(self):
        self.w = 0


    def fit(self, X, y):
        """
         learns the weights w in R^d according to the perceptron algorithm.
        :param X: in R^(m*d) - matrix of m samples in R^d
        :param y: in {+1,-1}^m -  classiﬁcations of the m samples
        :return:
        """
        m = y.shape[0]
        d = X.shape[1]
        t = 1
        w_t = np.zeros(d)
        exist_i = True

        i = 0
        while exist_i:
            if i < m:
                if y[i]*(np.dot(w_t,X[i])) <= 0: #todo : check if needs transpose
                    w_t = w_t + (y[i]*X[i])
                    i=0
                else:
                    i+=1

            else:
                exist_i = False

        self.w = w_t



    def predict(self, x):
        """
        predicts the label for a sample x using the classes current weight
        vector
        :param x:
        :return: +1 / -1
        """
        if np.inner(self.w ,x) > 0 :
            return 1
        else:
            return -1



def draw_and_classify_1(mean, covariance, m):

    # Draw training points {x1, . . . , xm} from a distribution D1
    x_train_1 = np.random.multivariate_normal(mean, covariance, m).T

    # classify the points according to a true hypothesis sign

    y_train_1 = classify_1(x_train_1)

    return  x_train_1, y_train_1


def classify_1(matrix):
    """
    Classiﬁy the points with f= sign<w, x> for
    :param matrix:
    :return:
    """

    size = matrix.shape[1]
    w =np.matrix([0.3, -0.5])
    w_matrix = np.repeat(w, size, axis=0)
    y_label = np.zeros(size)
    for i in range(size):
        y_label[i] = np.inner(w_matrix[i], matrix.T[i])
    y_label = np.sign(y_label)

    return y_label



def find_accuracy(x_train, y_train, x_test, y_test):
    """
    Train a Perceptron and SVM classiﬁer on {(xi, yi)}i=1 to m.
    and calculate accuracy
    :param:
    :return: accuracy
    """

    size = x_test.shape[1]

    per = Perceptron()
    svm = sv.SVC(C=1e10, kernel='linear')

    per.fit(x_train.T, y_train)
    svm.fit(x_train.T, y_train)

    y_predict_per = np.zeros(size)
    y_predict_svm = np.zeros(size)

    for i in range(size):
        y_predict_per[i] = per.predict(x_test.T[i])
        y_predict_svm[i] = svm.predict(x_test.T[i].reshape(1,-1))

    perceptron_accuracy = met.accuracy_score(y_test, y_predict_per)
    svm_accuracy = met.accuracy_score(y_test, y_predict_svm)

    return perceptron_accuracy, svm_accuracy


def draw_and_classify_2(n):

    y = np.random.binomial(1, p=0.5, size=n)
    y = np.where(y == 0, -1, 1)
    x = np.zeros((n,2))
    for i in range(n):
        if y[i] == 1:
            x[i] = np.array([np.random.uniform(-3, 1),np.random.uniform(1,3)])
        elif y[i] == -1:
            x[i] = np.array([np.random.uniform(-1, 3),np.random.uniform(-3,-1)])

    return x.T , y




def main():

    mean = np.zeros(2)
    covariance = np.identity(2)
    k= 10000

    #Draw test points {z1, . . . , zk} from the distribution D1

    x_test_1 = np.random.multivariate_normal(mean, covariance, k).T

    # calculate their true labels according to a true hypothesis sign

    y_test_1 = classify_1(x_test_1)

    # Draw test points {z1, . . . , zk} from the distribution D2

    x_test_2, y_test_2 = draw_and_classify_2(k)
    plt.show()


    perceptron_accuracy_for_m_1 = []
    perceptron_accuracy_for_m_2 = []
    svm_accuracy_for_m_1 = []
    svm_accuracy_for_m_2 = []

    for m in [5, 10, 15, 25, 70]:

        # save mean accuracies:

        mean_perceptron_accuracy_1, mean_perceptron_accuracy_2 = 0, 0
        mean_svm_accuracy_1,mean_svm_accuracy_2  = 0, 0


        for i in range(500):

            # D1

            # draw and classify:

            x_train_1, y_train_1 = draw_and_classify_1(mean, covariance, m)

            while np.all(y_train_1 == 1) or np.all(y_train_1 == -1):
                x_train_1, y_train_1 = draw_and_classify_1(mean, covariance, m)


            # update accuracies:

            perceptron_accuracy_1, svm_accuracy_1 = find_accuracy(
                x_train_1, y_train_1, x_test_1, y_test_1)

            mean_perceptron_accuracy_1 += perceptron_accuracy_1
            mean_svm_accuracy_1 += svm_accuracy_1

            # D2

            # draw and classify:

            x_train_2, y_train_2 = draw_and_classify_2(m)


            while np.all(y_train_2 == 1) or np.all(y_train_2 == -1):
                x_train_2, y_train_2 = draw_and_classify_2(m)

            # update accuracies:

            perceptron_accuracy_2, svm_accuracy_2 = find_accuracy(
                x_train_2, y_train_2, x_test_2, y_test_2)

            mean_perceptron_accuracy_2 += perceptron_accuracy_2
            mean_svm_accuracy_2 += svm_accuracy_2


        perceptron_accuracy_for_m_1 += [mean_perceptron_accuracy_1/500]
        svm_accuracy_for_m_1 += [mean_svm_accuracy_1/500]

        perceptron_accuracy_for_m_2 += [mean_perceptron_accuracy_2 / 500]
        svm_accuracy_for_m_2 += [mean_svm_accuracy_2 / 500]


    # hand in a plot of the mean accuracy vs. m
    m = [5, 10, 15, 25, 70]

    # plot for D1
    plt.subplot()
    plt.plot(m, perceptron_accuracy_for_m_1, label= 'perceptron', color='magenta')
    plt.plot(m, svm_accuracy_for_m_1, label='SVM', color='deepskyblue')
    plt.title('samples from $\mathcal{N}(0, \i_2)$')
    plt.legend(loc='best')
    plt.xlabel('m - number of train samples')
    plt.ylabel('mean accuracy')
    plt.savefig('d1.png')

    # plot for D2
    plt.subplot()
    plt.plot(m, perceptron_accuracy_for_m_2, label= 'perceptron', color='magenta')
    plt.plot(m, svm_accuracy_for_m_2, label='SVM', color='deepskyblue')
    plt.title('samples from rectangle uniform distribution')
    plt.legend(loc='best')
    plt.xlabel('m - number of train samples')
    plt.ylabel('mean accuracy')
    plt.savefig('d2.png')
