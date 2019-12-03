# Dataset 1: Wine Quality
# ABR: AdaBoost Regression
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.ensemble import AdaBoostRegressor
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

parameters={
    'n_estimators': scipy.stats.randint(50,200),
    'learning_rate': scipy.stats.reciprocal(0.1,2.0),
    'loss': ['linear', 'square', 'exponential']
}
abr=AdaBoostRegressor(random_state=0)
randcv = RandomizedSearchCV(abr, parameters, n_iter=100, verbose=1, random_state=0, cv=5)
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

parameters={
    'n_estimators': np.linspace(50,200, 10, dtype=np.int32),
    'learning_rate': scipy.stats.reciprocal(1.0, 3.0),
}
abr=AdaBoostRegressor(random_state=0)
randcv = RandomizedSearchCV(abr, parameters, n_iter=100, verbose=1, random_state=0, cv=5)
randcv.fit(X_train, y_train)

print(randcv.best_estimator_)
print(randcv.best_score_)
print(randcv.best_params_)

y_test_pred = randcv.best_estimator_.predict(X_test)
print(mean_squared_error(y_test, y_test_pred))