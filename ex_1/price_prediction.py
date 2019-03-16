import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# predict sale prices for houses

# Deﬁne y - the house prices.
# Deﬁne X the dataset, after your cleaning and pre-processing.
# We are looking for a vector w such that yˆ “ Xw is the best approximation for y.

def main():
    """

    :return:
    """
    x = pd.read_csv('../ex1/kc_house_data.csv')

    # drop columns
    x = x.drop(['id'], axis=1)

    # drop rows
    x = x.drop(x['long'].idxmax(), axis=0)
    x = x.drop(x['floors'].idxmin(), axis=0) # because he is the only one with 0 floors
    x = x.drop(x['bedrooms'].idxmax(), axis=0) # because of graph resolution and the ratio of sqft_living/rooms
    x = x.drop(x['price'].idxmin(), axis=0) # negative price

    #re-arrange date column
    x['date'] = x['date'].str.replace('T000000','')
    date = pd.to_numeric(x['date'])
    x = x.drop(['date'], axis=1)
    x=x.join(date)

    # ====== one hot decoding ======
    zipcode = pd.get_dummies(x['zipcode'])
    x = x.drop(['zipcode'], axis=1)
    x = x.join(zipcode)
    x=x.dropna()


    i = 1
    train_e = []
    test_e = []


    while i <= 100:

        # x_train , x_test = train_test_partition(x, i)

        x_train = x.sample(frac=i / 100, )
        x_test = x.drop(x_train.index, inplace=0)

        y_train = np.array([x_train['price']]).T
        just_x_train = x_train.drop(['price'], axis=1, inplace=0)
        x_train= np.array(just_x_train).T

        y_test = np.array([x_test['price']]).T
        just_x_test = x_test.drop(['price'], axis=1, inplace=0)
        x_test= np.array(just_x_test).T


        # train data-set over i
        w = least_squares(x_train, y_train)

        # Training error is the error that you get when you run the trained
        #  model back on the training data.
        train_sol = x_train.T@w
        train_error = root_mean_square_error(y_train, train_sol)
        train_e += [train_error]

        # Test error is the error when you get when you run the trained model
        #  on a set of data that it has previously never been exposed to.
        test_sol = x_test.T@w
        test_error = root_mean_square_error(y_test, test_sol)
        test_e += [test_error]

        i+=1

    # Plot a graph of train and test error as a function of x

    a , b= plt.subplots()
    plt.title("train and test error as a function of x")
    plt.plot(train_e, label='train error')
    plt.plot(test_e, label='test error')
    plt.legend(loc='best')
    plt.show()

    lis = list(range(10,100))
    a , b= plt.subplots()
    plt.title("train and test error as a function of x")
    plt.plot(lis,train_e[10:], label='train error')
    plt.plot(lis, test_e[10:], label='test error')
    plt.legend(loc='best')
    plt.show()


def least_squares(x, y):
    """
    Solves the equation xw=y by computing a vector w that
    minimizes the Euclidean 2-norm || y - xw ||^2
    :param x: np array
    :param y: np array
    :return: w: least-squares solution
    """
    # check if x is singular - check if the rank of x = num of columns
    is_singular = True
    r = np.linalg.matrix_rank(x)
    if r == x.shape[0]:
        is_singular = False

    if is_singular:
        # x = usv^t svd of x
        u, s_v, vt = np.linalg.svd(x)
        s = np.zeros(x.shape)
        s[:len(s_v), :len(s_v)] = np.diag(s_v)
        # xx^t = udu^t evd of
        d = s@s.T

        # get d knife
        d_knife = np.zeros(d.shape)
        d_d = 1/np.diag(d[:len(s_v),:len(s_v)])
        d_knife[:len(s_v),:len(s_v)] = np.diag(d_d)

        w = u @ d_knife @ s @ vt  @ y

    else:

        # get xx^t
        x_x_t = x@x.T
        # get xy
        x_y = x@y
        # solve xx^tw = xy - the normal equation
        w = np.linalg.solve(x_x_t, x_y)

    return w


def root_mean_square_error(solution , prediction):
    """
    ||solution - prediction||^2
    :param solution:
    :param prediction:
    :return:
    """
    return np.mean((solution - prediction)**2)**(0.5)

main()