# Dataset 1: Wine Quality
# GPR: Gaussian Process Regression
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV

data = np.loadtxt('Datasets/winequality-red.csv', delimiter=';', skiprows=1)
X=data[:,:11]
y=data[:,11:].astype(np.int32).reshape(-1,)

scaler = sklearn.preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1,random_state=0)

#parameters = {'max_depth':range(1, 6)}
parameters = {}
gpr=GaussianProcessRegressor(random_state=0)
gs=GridSearchCV(gpr, parameters, cv=3, verbose=1)
gs.fit(X_train,y_train)
print(gs.best_estimator_)