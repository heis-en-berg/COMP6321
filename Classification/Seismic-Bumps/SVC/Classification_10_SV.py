# Dataset 10: Seismic-Bumps
# SV: Support Vector Classification
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
from sklearn.metrics import accuracy_score

data = np.loadtxt('seismic-bumps.arff', delimiter=',',skiprows=154,dtype='str')
X_old=data[:,:18]
y=data[:,18:]

letters = {'a':'1','b':'2','c':'3','d':'4','W':'1','N':'2'}
X=np.copy(X_old)
for old, new in letters.items():
    X[X_old==old] = new

X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.2,random_state=0)
scaler=sklearn.preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

parameters = {
    'C'     : scipy.stats.reciprocal(1.0, 1000.),
    'kernel' : ["poly", "rbf"],
    'degree' : np.linspace(1,10,5, dtype=np.int32),
    'gamma' : np.linspace(1,15,10, dtype=np.int32),
    'coef0': [0,0.1,1,2,3],
    'max_iter' : [20000],
    'cache_size' : [1000]
}

svc=SVC(random_state=0)
randcv = RandomizedSearchCV(svc, parameters, n_iter=100, verbose=1, random_state=0, cv=5)
randcv.fit(X_train, y_train)

print(randcv.best_estimator_)
print(randcv.best_score_)
print(randcv.best_params_)

y_test_pred = randcv.best_estimator_.predict(X_test)
print(accuracy_score(y_test, y_test_pred))