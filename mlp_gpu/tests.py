import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from mlp_gpu import MLPClassifierGPU


def test_iris():
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
    acc_sklearn = test_model_sklearn(X_train, y_train, X_test, y_test)
    print('acc_sklearn = ', acc_sklearn)
    acc_custom = test_model_custom(X_train, y_train, X_test, y_test)
    print('acc_custom = ', acc_custom)


def test_wine():
    X, y = sklearn.datasets.load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
    acc_sklearn = test_model_sklearn(X_train, y_train, X_test, y_test)
    print('acc_sklearn = ', acc_sklearn)
    acc_custom = test_model_custom(X_train, y_train, X_test, y_test)
    print('acc_custom = ', acc_custom)


def test_big_random():
    X = np.random.randn(10000, 20)
    y = np.random.randint(low=0, high=2, size=10000)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
    acc_sklearn = test_model_sklearn(X_train, y_train, X_test, y_test)
    print('acc_sklearn = ', acc_sklearn)
    acc_custom = test_model_custom(X_train, y_train, X_test, y_test)
    print('acc_custom = ', acc_custom)


def test_bigger_random():
    X = np.random.randn(10000, 20)
    y = np.random.randint(low=0, high=2, size=10000)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
    acc_sklearn = test_model_sklearn(X_train, y_train, X_test, y_test)
    print('acc_sklearn = ', acc_sklearn)
    acc_custom = test_model_custom(X_train, y_train, X_test, y_test)
    print('acc_custom = ', acc_custom)


def timeit(func):
    def new_func(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f'exection of {func.__name__} finished, time = {end - start}')
        return res
    return new_func


@timeit
def test_model_sklearn(X_train, y_train, X_test, y_test):
    mlp_sklearn = MLPClassifier()
    mlp_sklearn.fit(X_train, y_train)
    pred_sklearn = mlp_sklearn.predict(X_test)
    acc_sklearn = sklearn.metrics.accuracy_score(y_test, pred_sklearn)
    return acc_sklearn


@timeit
def test_model_custom(X_train, y_train, X_test, y_test):
    mlp_custom = MLPClassifierGPU()
    mlp_custom.fit(X_train, y_train)
    pred_custom = mlp_custom.predict(X_test)
    acc_custom = sklearn.metrics.accuracy_score(y_test, pred_custom)
    return acc_custom


if __name__ == '__main__':
    test_iris()
    test_wine()
    test_big_random()
    test_bigger_random()
