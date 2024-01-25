from matplotlib.pyplot import sca
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, Ridge,ElasticNet,Lasso
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,mean_squared_log_error

from .util import rsquared
np.seterr(divide='ignore', invalid='ignore')

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

    def scaler_fit(self,X,y):
        scaler_X = self.scaler.fit_transform(np.hstack((np.array(X),np.array(y).reshape(-1,1))))
        # scaler_X = np.vstack((scaler_X,np.zeros((len(scaler_X),len(scaler_X[0])))))
        self.model.fit(scaler_X[:,:-1],scaler_X[:,-1])
        coefs = []
        for i in range(len(self.model.coef_)):
            if (self.scaler.data_max_[i]-self.scaler.data_min_[i])!=0:
                coef = self.model.coef_[i]*(self.scaler.data_max_[-1]-self.scaler.data_min_[-1])/(self.scaler.data_max_[i]-self.scaler.data_min_[i])
                coefs.append(coef)
            else:
                coefs.append(0)
        b = [coefs[i]*self.scaler.data_min_[i] for i in range(len(coefs))]
        intercept = self.model.intercept_*(self.scaler.data_max_[-1]-self.scaler.data_min_[-1]) - sum(b) + self.scaler.data_min_[-1]

        self.coef_ = coefs
        self.intercept_ = intercept


    def fit(self,X,y):
        # lr = LinearRegression(fit_intercept=False,positive=True)
        # lr.fit(X,y,sample_weight=y)
        # lr = Ridge(alpha=0.1, max_iter=30000, positive=True, fit_intercept=False)
        # lr.fit(X, y, sample_weight=np.log(1.0 / y.astype(float) + 1000))
        # lr = HuberRegressor(fit_intercept = False,)
        if self.type == "ransac":
            self.model = RANSACRegressor(base_estimator=LinearRegression(fit_intercept=False, positive=True),min_samples=round(len(X)/2))
        # if self.type == "ridge":
        #     self.scaler_fit(X,y)
        #     return
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

