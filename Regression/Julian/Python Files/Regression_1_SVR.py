# Dataset 1: Wine Quality
# SVR: Support Vector Regression
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

data = np.loadtxt('Datasets/winequality-red.csv', delimiter=';', skiprows=1)
X=data[:,:11]
y=data[:,11:].astype(np.int32).reshape(-1,)

scaler = sklearn.preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1,random_state=0)

parameters = {'C':np.logspace(0, 2, num=3),'epsilon':np.logspace(-2, 0, num=3),'gamma':np.logspace(-2, 0, num=3)}
svr=SVR(kernel='rbf')
gs=GridSearchCV(svr, parameters, cv=3, verbose=1)
gs.fit(X_train,y_train)
print(gs.best_estimator_)
#print(gs.get_params())


C=np.logspace(0, 2, num=3)
epsilon=np.logspace(-2, 0, num=3)
gamma=np.logspace(-2, 0, num=3)
print(C,epsilon,gamma)
for i in range(C.shape[0]):
    for j in range(epsilon.shape[0]):
        for k in range(gamma.shape[0]):
            svr = SVR(kernel='rbf',C=C[i],epsilon=epsilon[j],gamma=gamma[k])
            svr.fit(X_train,y_train)
            print(svr.score(X_train,y_train)*100,"     C:",C[i],"     epsilon:",epsilon[j],"     gamma:",gamma[k])