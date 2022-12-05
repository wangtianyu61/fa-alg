from sklearn.linear_model._base import LinearModel
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y
import numpy as np
import pandas as pd
import time
import cvxpy as cp

class MinmaxLinearModel(LinearModel, RegressorMixin):
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, nonnegative=False, tol=1e-15):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.nonnegative = nonnegative
        self.tol = tol

    def fit(self, X, y, group, min_coef = None, max_coef = None):
        group_list = list(set(group))
        group_num = len(set(group))
        # split (X, y) into G groups
        X, y = check_X_y(X, y, accept_sparse = ['csr', 'csc', 'coo'], y_numeric = True,
                            multi_output = False)
        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize, copy=self.copy_X)
        ## split them into {X[j], y[j]}_{j = 1}^G
        
        X_org = []
        y_org = []
        num_in_group = np.zeros(group_num)
        for j, group_val in enumerate(group_list):
            
            group_index = [i for i in range(len(group)) if group[i] == group_val]
            num_in_group[j] = len(group_index)
            X_org.append(X[group_index])
            y_org.append(y[group_index])

        w = cp.Variable(X.shape[1])
        b = cp.Variable()
        aux_t = cp.Variable()
        obj = cp.Minimize(aux_t)
        constraint = [aux_t >= 0]
        for j in range(group_num):
            constraint += [cp.sum_squares(X_org[j] @ w + b - y_org[j]) <= aux_t * num_in_group[j]]
        problem = cp.Problem(obj, constraint)
        problem.solve(solver = cp.GUROBI)
        self.coef_ = w.value
        self.intercept_ = b.value


            