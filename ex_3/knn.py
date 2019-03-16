import numpy as np
import sklearn.linear_model as lm
import  urllib.request as ur
import  matplotlib.pyplot as plt
import sklearn.neighbors as nei


class KNN:
    """
    this class implements a k-nearest neighbors classiﬁer
    """

    def __init__(self, k):
        """

        :param k:determines the number of nearest neighbors for the classiﬁer.
        """
        self.num_of_nei = k
        self.classifier = nei.KNeighborsClassifier(n_neighbors=k)

    def fit(self, X, y):
        """
        simply stores the data
        :param X:
        :param y:
        :return:
        """
        self.classifier.fit(X, y)

    def predict(self, x):
        """
        predicts x’s label according to the majority of its k nearest neighbors
        :param x:
        :return:
        """
        return self.classifier.predict(x)





def get_current_N_i (i, labeled, sorted_prediction_i):
    """

    :param i:
    :param prediction:
    :param sorted_prediction_i:
    :return:
    """

    index, counter, threshold = 0, 0 ,0
    while counter != i:
        real_i = sorted_prediction_i[index]
        if labeled[real_i] == 1:
            counter+=1
        threshold+=1
        index+=1
    return threshold


def main():
    """
    main for q8
    :return:
    """

    points = [1, 2, 5, 10, 100]

    # a - Draw 1000 data points from the dataset and keep them aside as a test set.


    url = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data"
    raw_data = ur.urlopen(url)
    data = np.loadtxt(raw_data, delimiter=" ")

    np.random.shuffle(data)
    test_set = data[:1000]
    train_set = data[1000:]

    train_set_x = train_set[:,:57]
    train_set_y = train_set[:,57]
    test_set_x = test_set[:,:57]
    test_set_y = test_set[:,57]


    # b - Fit a Logistic Regression model on the rest of the data

    logistic_re = lm.LogisticRegression()
    logistic_re.fit(train_set_x,train_set_y)

    # Use predict_proba on the test set and sort according to the
    # probability of the classiﬁer to predict y = 1

    test_set_prediction = logistic_re.predict_proba(test_set_x).T[1]
    sorted_prediction_index = np.argsort(test_set_prediction)[::-1]


    # NP = the number of test samples whose true label is 1
    NP = test_set_y.sum().astype('int')
    NN = len(test_set_y) - NP
    TPR = np.zeros(NP+1)
    FPR = np.zeros(NP+1)


    # the loop checks for each i how many samples we need to tag with
    #  label 1 in order to get TPR = i/NP
    for i in range(NP):
        curr_tpr = i / NP
        N_i = get_current_N_i(i, test_set_y, sorted_prediction_index)
        curr_fpr = (N_i - i) / NN
        TPR[i] = curr_tpr
        FPR[i] = curr_fpr
    TPR[NP] =1
    FPR[NP] =1


    plt.plot(FPR, TPR ,color='magenta')
    plt.title('ROC curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig('roc.png')
    plt.show()


    # c
    error_for_k = np.zeros(len(points))
    mean_for_k = np.zeros(len(points))


    for j in range(10):
        for i in range(len(points)):
            k = points[i]
            knn = KNN(k)
            knn.fit(train_set_x, train_set_y)
            knn_predict_y = knn.predict(test_set_x)
            for l in range(len(knn_predict_y)):
                if knn_predict_y[l] != test_set_y[l]:
                    error_for_k[i] +=1

    for i in range(len(points)):
        mean_for_k[i] = error_for_k[i]/ (10*1000)

    plt.plot(points, mean_for_k, label = 'knn' ,color='magenta')
    plt.title('samples from spam.data')
    plt.legend(loc='best')
    plt.xlabel('k - number of nearest neighbors for the classiﬁer')
    plt.ylabel('mean error')
    plt.savefig('knn.png')





















main()






