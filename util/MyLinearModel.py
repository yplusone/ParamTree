from matplotlib.pyplot import sca
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, Ridge,ElasticNet,Lasso
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,mean_squared_log_error

from .util import rsquared

class MyLinearRegression():
    def __init__(self,type = "ridge"):
        self.type = type
        if type == "linear":
            self.model = LinearRegression(fit_intercept=False,positive=True,normalize=True)
        elif type == "ridge":
            self.model = Ridge(alpha=1,positive=True, fit_intercept=False)
        elif type == "ransac":
            self.model = RANSACRegressor(base_estimator=LinearRegression(fit_intercept=False, positive=True,normalize=True))
        elif type == "huber":
            self.model = HuberRegressor(fit_intercept = False,)
        elif type == "ElasticNet":
            self.model = ElasticNet(alpha=10,fit_intercept=True,normalize=True,positive=True)
        elif type == "Lasso":
            self.model = Lasso(alpha=100,fit_intercept=True,normalize=True,positive=True)
        else:
            raise Exception("Wrong regression type")
        self.scaler = MinMaxScaler()


    def fit(self,X,y):
        if self.type == "ransac":
            self.model = RANSACRegressor(base_estimator=LinearRegression(fit_intercept=False, positive=True),min_samples=round(len(X)/2))
        lr = self.model
        lr.fit(X, y.reshape(-1,1))
        if self.type in ["ransac"]:
            intercept = lr.estimator_.intercept_
            coef = lr.estimator_.coef_
        else:
            intercept = lr.intercept_
            coef = lr.coef_
        if type(coef[0].tolist())== float:
            self.coef_ = coef.tolist()
        else:
            self.coef_ = coef[0].tolist()
        self.intercept_ = intercept

    def predict(self,X):
        theta = deepcopy(self.coef_)
        theta.append(self.intercept_)
        X_n = np.hstack((np.array(X),np.ones(len(X)).reshape(-1,1)))
        return X_n.dot(np.array(theta).reshape(-1,1)).reshape(-1)

    def score(self,X,y):
        y_predict = self.predict(X)
        return rsquared(y_predict,y)

    def mape_score(self,X,y):
        y_predict = self.predict(X)
        return mean_absolute_percentage_error(y_predict, y)

