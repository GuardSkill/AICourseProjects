from numpy import genfromtxt
import numpy as np
import math
from model import model
from model import one_hot_matrix
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def random_dataset(X, Y, seed=0):
    m = X.shape[0]  # number of training examples
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case
    return shuffled_X, shuffled_Y


def get_high_dimension(X):
    # numeraical_rows = [0, 3, 4, 7, 9, 12]
    numeraical_colmns = [3, 4, 7]
    temp_X = np.array([], dtype=np.float64).reshape(X.shape[0], 0)
    for x1 in numeraical_colmns:
        for x2 in numeraical_colmns:
            temp = X[:, x1] * X[:, x2]
            # temp_X = np.concatenate([temp_X, temp], axis=1)
            temp_X = np.hstack([temp_X, temp])
    return np.concatenate([X, temp_X], axis=1)


def neural_method(X_train, Y_train, X_test, Y_test, message):
    parameters, precision = model(
        X_train.transpose(), Y_train.transpose(), X_test.transpose(), Y_test.transpose(), message,
        learning_rate=0.0015, num_epochs=10000, print_cost=True)
    return precision


def load_data(type_str):
    if type_str == 'heart':
        noise_data = genfromtxt('data/processed.cleveland.data', delimiter=',')
        data = noise_data[~np.isnan(noise_data).any(axis=1)]
        X = data[:, :-1]
        Y = one_hot_matrix(data[:, -1], 5)
    elif type_str == 'shuttle':
        data = genfromtxt('data/shuttle.tst', delimiter=' ')
        Y = data[:, -1]
        Y[Y == 1] = 0
        Y[Y == 4] = 1
        Y[Y == 5] = 2
        X = data[:, :-1]
        Y = one_hot_matrix(Y, 3)
    elif type_str == 'iris':
        # data = genfromtxt('iris.data', delimiter=',')
        data_df = pd.read_csv('data/iris.data', delimiter=',', header=None)
        data_df.iloc[[data_df.iloc[:, -1] == 'Iris-setosa'], [data_df.columns[-1]]] = 0
        data_df.iloc[[data_df.iloc[:, -1] == 'Iris-versicolor'], [data_df.columns[-1]]] = 1
        data_df.iloc[[data_df.iloc[:, -1] == 'Iris-virginica'], [data_df.columns[-1]]] = 2

        # for i in data_df.__len__():
        X = np.array(data_df.iloc[:, :-1])
        Y = one_hot_matrix(np.array(data_df.iloc[:, -1]), 3)

    return X, Y


def SVM_method(X_train, Y_train, X_test, Y_test, message):
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, Y_train.squeeze())
    Y_pred = svclassifier.predict(X_test)
    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))
    return None


data_name = 'iris'
X, Y = load_data(data_name)
test_precision = np.ones((10, 10))
# columns=['%d fold' % x for x in range(1,11)]
# rows = ['%d times' % x for x in range(1,11)]
for i in range(10):  # ten times
    X, Y = random_dataset(X, Y, seed=i)  # i+1 time random shuffle
    for j in range(10):  # ten-fold
        test_start_pos = math.floor(j * X.shape[0] / 10)  # divide dataset to 10 components
        test_over_pos = math.floor((j + 1) * X.shape[0] / 10)
        X_test = X[test_start_pos:test_over_pos, :]
        Y_test = Y[test_start_pos:test_over_pos, :]
        X_train = np.delete(X, slice(test_start_pos, test_over_pos), axis=0)
        Y_train = np.delete(Y, slice(test_start_pos, test_over_pos), axis=0)
        precision = neural_method(X_train, Y_train, X_test, Y_test, str(i + 1) + "th times " + str(j + 1) + "th fold")
        test_precision[i][j] = precision
np.savetxt('result/'+data_name + '.csv', test_precision, delimiter=',')
print("Ave_Precision:", np.mean(test_precision))
