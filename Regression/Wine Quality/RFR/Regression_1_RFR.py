# Dataset 1: Wine Quality
# RFR: Random Forest Regression
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
from sklearn.metrics import mean_squared_error

#RED WINE
data = np.loadtxt('winequality-red.csv', delimiter=';', skiprows=1)
X=data[:,:11]
y=data[:,11:].reshape(-1,)

X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.2,random_state=0)
scaler=sklearn.preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

parameters = {
        'n_estimators': [10,50,100,150],
        'criterion': ['mse', 'mae'],
        'max_depth': [10,20,30,50]
        }
rfr=RandomForestRegressor(random_state=0)
randcv = RandomizedSearchCV(rfr, parameters, n_iter=500, verbose=1, random_state=0, cv=10)
randcv.fit(X_train, y_train)

print(randcv.best_estimator_)
print(randcv.best_score_)
print(randcv.best_params_)

y_test_pred = randcv.best_estimator_.predict(X_test)
print(mean_squared_error(y_test, y_test_pred))

#WHITE WINE
data = np.loadtxt('winequality-white.csv', delimiter=';', skiprows=1)
X=data[:,:11]
y=data[:,11:].reshape(-1,)

X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.2,random_state=0)
scaler=sklearn.preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

parameters = {
        'n_estimators': [10,50,100,150],
        'criterion': ['mse', 'mae'],
        'max_depth': [10,20,30,50]
        }
rfr=RandomForestRegressor(random_state=0)
randcv = RandomizedSearchCV(rfr, parameters, n_iter=500, verbose=1, random_state=0, cv=10)
randcv.fit(X_train, y_train)

print(randcv.best_estimator_)
print(randcv.best_score_)
print(randcv.best_params_)

y_test_pred = randcv.best_estimator_.predict(X_test)
print(mean_squared_error(y_test, y_test_pred))