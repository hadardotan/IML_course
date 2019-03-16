import numpy as np
import  matplotlib.pyplot as plt

NUM_OF_SEQUENCES = 100000
NUM_OF_TOSSES = 1000
P = 0.25

def main():
    data = np.random.binomial(1, 0.25, (100000, 1000))
    epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]

    # A
    x_ms = np.empty([100000, 1000])
    fig, ax = plt.subplots()
    for i in range(NUM_OF_SEQUENCES):
        for m in range(NUM_OF_TOSSES):
            x_ms[i,m] = np.mean(data[i, 0:m + 1])
        # plot a for 0<=i<5
        if i in range(5):
            plt.plot(x_ms[i], label='$ row={i}$'.format(i=i+1))
        if i == 4:
            plt.title("mean estimate of all tosses up to m as function of m ")
            plt.legend(loc='best')
            plt.xlabel('m')
            plt.ylabel('mean of tosses up to m')
            plt.show()

    for e in epsilon:
        # B
        plt.subplots()
        bounds = ['Chebyshev', 'Hoeﬀding']
        cheby = []
        hoff = []
        for m in range(1,NUM_OF_TOSSES+1):
            for bound in bounds:
                if bound == 'Chebyshev':
                    chebb = 1/(4*m*e*e)
                    if chebb >1:
                        cheby += [1]
                    else:
                        cheby += [chebb]

                    cheby += []
                elif bound == 'Hoeﬀding':
                    hoffi = 2*np.exp(-2*m*e*e)
                    if hoffi >1:
                        hoff += [1]
                    else:
                        hoff += [hoffi]

        plt.plot(cheby, label='Chebyshev')
        plt.plot(hoff, label='Hoeﬀding')
        plt.legend(loc='best')

        plt.title("upper bound up to m coin tosses as function of m for "
                  "epsilon=" + str(e))
        plt.xlabel('m')
        plt.ylabel('upper bound')
        plt.show()

        # C
        # percentage of sequences that satisfy |X_m - E[X]| >= e as function
        #  of m

        epsilon_matrix = np.full((100000, 1000), e)
        satisfy = np.absolute(x_ms - P)
        bool_for_satisfy = np.greater_equal(satisfy, epsilon_matrix)
        count_num_for_m = bool_for_satisfy.sum(axis=0)
        count_num_for_m = count_num_for_m/NUM_OF_SEQUENCES

        plt.plot(count_num_for_m )
        plt.title('$percentage of sequences that satisfy for e= {e} as function'
                  ' of m$'.format(e=e))
        plt.show()


main()