import numpy as np
import matplotlib.pyplot as plt
from uids.utils.DataAnalysis import *
from uids.utils.lof import LocalOutlierFactor


# ================================= #
#        Test Functions

def test_lof():
    # samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
    # neigh = NearestNeighbors(n_neighbors=1)
    # neigh.fit(samples)

    np.random.seed(42)

    # Generate train data
    X = 0.3 * np.random.randn(100, 2)
    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    X = np.r_[X + 2, X - 2, X_outliers]

    # fit the model
    clf = LocalOutlierFactor(n_neighbors=20)
    y_pred = clf.fit_predict(X)
    y_pred_outliers = y_pred[200:]

    # plot the level sets of the decision function
    xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("Local Outlier Factor (LOF)")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

    a = plt.scatter(X[:200, 0], X[:200, 1], c='white')
    b = plt.scatter(X[200:, 0], X[200:, 1], c='red')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([a, b],
               ["normal observations",
                "abnormal observations"],
               loc="upper left")
    plt.show()

# ================================= #
#              Main

if __name__ == '__main__':
    # test_ABOD_3()
    test_lof()
