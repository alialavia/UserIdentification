import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from uids.v2.set_metrics import *
from sklearn.exceptions import NotFittedError


class ABODEstimator(BaseEstimator):
    """
    Parameters
    ----------
    T : float, optional
    """

    # reference data
    __tmp_data = None
    # classification threshold
    T = 0

    def __init__(self, T=0.3):
        self.T = T

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        Returns
        -------
        self : object
            Returns self.
        """
        self.__tmp_data = X
        return self

    def decision_function(self, X):
        return ABOD.get_score(X, self.__tmp_data)

    def predict(self, X):
        """
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
        """

        # calc abod score
        abod_score = self.decision_function(X)

        # threshold
        decision = np.array([1]*len(X))
        decision[abod_score < self.T] = -1
        return decision


class L2Estimator(BaseEstimator):
    """
    Parameters
    ----------
    T : float, optional
    """

    # reference data
    __tmp_data = None
    # classification threshold
    T = 0
    # mean_ref or mean
    comparison = None

    def __init__(self, T=0.99, comparison='mean_ref'):
        self.T = T
        self.comparison = comparison

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        Returns
        -------
        self : object
            Returns self.
        """
        self.__tmp_data = X
        return self

    def decision_function(self, X):
        if self.__tmp_data is None:
            raise NotFittedError("Estimator not fitted, call `fit` before"
                                 " `feature_importances_`.")

        if self.comparison == 'mean_ref':
            l2_score = pairwise_distances(np.average(self.__tmp_data, axis=0).reshape(1, -1), X,
                                          metric='euclidean')[0]
            l2_score = np.square(l2_score)
            return l2_score
        else:
            raise ValueError("'{}' is not a valid comparison method".format(self.comparison))

    def predict(self, X):
        """
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
        """

        if self.__tmp_data is None:
            raise NotFittedError("Estimator not fitted, call `fit` before"
                                 " `feature_importances_`.")

        l2_score = self.decision_function(X)

        # threshold
        decision = np.array([1]*len(X))
        decision[l2_score > self.T] = -1
        return decision


class CosineDistEstimator(BaseEstimator):
    """
    Parameters
    ----------
    T : float, optional
    """

    # reference data
    __tmp_data = None
    # classification threshold
    T = 0
    # mean_ref or mean
    comparison = None

    def __init__(self, T=0.7, comparison='mean_ref'):
        self.T = T
        self.comparison = comparison

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        Returns
        -------
        self : object
            Returns self.
        """
        self.__tmp_data = X
        return self

    def decision_function(self, X):
        if self.comparison == 'mean_ref':
            cosine_score = pairwise_distances(np.average(self.__tmp_data, axis=0).reshape(1, -1), X,
                                          metric='cosine')[0]
            return cosine_score
        else:
            raise ValueError("'{}' is not a valid comparison method".format(self.comparison))

    def predict(self, X):
        """
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
        """

        if self.__tmp_data is None:
            raise NotFittedError("Estimator not fitted, call `fit` before"
                                 " `feature_importances_`.")

        cosine_score = self.decision_function(X)

        # threshold
        decision = np.array([1]*len(X))
        decision[cosine_score > self.T] = -1
        return decision