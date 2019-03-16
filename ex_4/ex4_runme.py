"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex4.

Author: Hadar Dotan
Date: May, 2018

"""

import numpy as np
import matplotlib.pyplot as plt
import adaboost
import decision_tree
import ex4_tools as tools
from sklearn.model_selection import KFold


# helper functions



def get_syn_data():

    syn_data_files  = ['X_test.txt',
                       'X_train.txt',
                       'X_val.txt',
                       'y_test.txt',
                       'y_train.txt',
                       'y_val.txt']

    syn_data = [np.loadtxt('SynData/'+file_name) for file_name in syn_data_files]
    return syn_data

def plot_decisions(relative_range, learned_classifiers, train_X, train_y, name):
    for i, t in enumerate(relative_range):
        tools.decision_boundaries(learned_classifiers[i], train_X, train_y,
                            "{name} decisions for T={t}".format(name=name, t=t))

def Q3(): # AdaBoost
    """

    :return:
    """

    syn_data = get_syn_data()
    X_test, X_train, X_val, y_test, y_train, y_val = syn_data[0], syn_data[1],\
                                                   syn_data[2], syn_data[3],\
                                                   syn_data[4], syn_data[5]

    # init T
    T = 22*[0]
    T[0] = 1
    for i in range(20):
        T[i+1] = (i+1)*5

    T[21] = 200
    keep_T =[1, 5, 10, 50, 100, 200]
    learned_classifiers = [None] * len(keep_T)
    min_val_err = 1
    min_val_err_classifier = None
    training_error, validation_error, test_error = [], [], []

    for t in T:


        ada_boost = adaboost.AdaBoost(WL=tools.DecisionStump, T=t)
        ada_boost.train(X_train, y_train)

        if t in keep_T:
            learned_classifiers[keep_T.index(t)] = ada_boost

        if t!=1:
            training_error.append(ada_boost.error(X_train, y_train))
            validation_error.append(ada_boost.error(X_val, y_val))


    plot_decisions(keep_T, learned_classifiers, X_train, y_train, "adaBoost on SynData")
    plt.plot(T[1:], training_error, label='training error', color='magenta')
    plt.plot(T[1:], validation_error, label='validation error',color='deepskyblue')
    plt.title('adaBoost error on SynData as function of T')
    plt.legend(loc='best')
    plt.xlabel('T - number of base learners to learn')
    plt.ylabel('Error')
    plt.show()


def Q4(): # decision trees

    syn_data = get_syn_data()
    X_test, X_train, X_val, y_test, y_train, y_val = syn_data[0], syn_data[1], \
                                                     syn_data[2], syn_data[3], \
                                                     syn_data[4], syn_data[5]

    D = [3, 6, 8, 10, 12]
    training_error, validation_error = [], []
    learned_classifiers = [None] * len(D)

    for d in D:
        dt = decision_tree.DecisionTree(d)
        dt.train(X_train, y_train)

        learned_classifiers[D.index(d)] = dt

        training_error.append(dt.error(X_train, y_train))
        validation_error.append(dt.error(X_val, y_val))


    plot_decisions(D, learned_classifiers, X_train, y_train, "CART DT on SynData")
    plt.plot(D, training_error, label='training error', color='magenta')
    plt.plot(D, validation_error, label='validation error',
             color='deepskyblue')
    plt.title('CART DT error on SynData as function of max depth')
    plt.legend(loc='best')
    plt.xlabel('Max Depth')
    plt.ylabel('Error')
    plt.show()


def get_spam_data(size=1536):
    """
    Read the spam dataset, sample a test set of size 1536 and place it in a
    data vault.
    :return:
    """


def split_data_to_folds(data):
    data_size = len(data)
    data_sets = [0]*data_size
    for i in range(data_size):
        test = np.vstack([data[j] for j in range(len(data)) if j != i])
        validation = data[i]
        data_sets[i] = np.array([test, validation])

    return data_sets

def Q5(): # spam data

    T = [5, 50, 100, 200, 500, 1000]
    D = [5, 8, 10, 12, 15, 18]
    # get spam data
    spam_data = np.loadtxt('SpamData/spam.data')

    # change values of 0 to -1
    spam_data[:, -1][spam_data[:, -1] == 0] = -1\

    # get vault data and train data
    np.random.shuffle(spam_data)
    vault_index = np.random.choice(len(spam_data), 1536, replace=False)
    train_index = np.array([i for i in range(len(spam_data)) if i not in vault_index])
    train_data = spam_data[train_index]

    vault_data = spam_data[vault_index]

    # Use 5-fold cross validation to pick T and d
    data_size = len(train_data)
    split = int(data_size / 5)
    folds = np.split(train_data, [split, 2 * split, 3 * split, 4 * split])
    data_sets = split_data_to_folds(folds)


    DT_error = [0]*6
    adaboost_error = [0]*6
    best_DT_error = None
    bes_adaboost_error = None


    for i in range(5):

        fold_size1 = data_sets[i][0].shape[1]

        arr1 = data_sets[i][0]
        arr2 = data_sets[i][1]



        X_train, y_train = arr1[:, 0:fold_size1 - 1],\
                           arr1[:,fold_size1 - 1:fold_size1]


        X_validation = arr2[:,0:(fold_size1-1)]
        y_validation = arr2[:,(fold_size1 - 1):fold_size1]


        y_train = y_train.reshape((-1,))
        y_validation = y_validation.reshape((-1,))


        for t in T:
            ada_boost = adaboost.AdaBoost(tools.DecisionStump,t)
            ada_boost.train(X_train,y_train)
            current_adaboost_error = ada_boost.error(X_validation, y_validation)
            adaboost_error[i] += current_adaboost_error
            if bes_adaboost_error == None or bes_adaboost_error > current_adaboost_error:
                bes_adaboost_error = t



        for d in D:
            dt = decision_tree.DecisionTree(d)
            dt.train(X_train, y_train)
            current_dt_error  = dt.error(X_validation, y_validation)
            DT_error[i] += current_dt_error
            if best_DT_error == None or best_DT_error > current_dt_error:
                best_DT_error = d


    # get mean error
    adaboost_error = np.array([x/5 for x in adaboost_error])
    DT_error = np.array([x/5 for x in DT_error])

    plt.errorbar(T, adaboost_error,capsize=np.std, color='magenta')
    plt.title('validation error on SpamData for adaBoost as function of T')
    plt.legend(loc='best')
    plt.xlabel('T')
    plt.ylabel('Error')
    plt.errorbar(T, adaboost_error)
    plt.show()

    #
    plt.errorbar(D, DT_error,capsize=np.std, color='magenta')
    plt.title('validation error on SpamData for DT as function of max depth')
    plt.legend(loc='best')
    plt.xlabel('max depth')
    plt.ylabel('Error')
    plt.show()



    # Train classifiers using the chosen parameter values, using the complete training set.
    #


    X_train, y_train = train_data[:,0:57], train_data[:,57]
    X_vault, y_vault = vault_data[:,0:57],vault_data[:,57]


    ada_boost = adaboost.AdaBoost(tools.DecisionStump,bes_adaboost_error)
    ada_boost.train(X_train,y_train)
    vault_adaboost_error = ada_boost.error(X_vault, y_vault)


    dt = decision_tree.DecisionTree(best_DT_error)
    dt.train(X_train,y_train)
    vault_dt_error = dt.error(X_vault, y_vault)


    print("vault_adaboost_error= "+vault_adaboost_error)
    print("vault_dt_error= "+vault_dt_error)


if __name__ == '__main__':
    # TODO - run your code for questions 3-5
    pass



Q5()