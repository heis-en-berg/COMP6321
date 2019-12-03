# Dataset 9: Thoracic Surgery Data
# LR: Logistic Regression Classification
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
from sklearn.metrics import accuracy_score

data = np.loadtxt('ThoraricSurgery.arff', delimiter=',',dtype='str',skiprows=21)
X_old=data[:,:16]
y_old=data[:,16:]
X_old=np.char.strip(X_old)
y_old=np.char.strip(y_old)

dictionary = {'DGN3':'1','DGN2':'2','DGN4':'3','DGN6':'4',
              'DGN5':'5','DGN8':'6','DGN1':'7','PRZ2':'1',
              'PRZ1':'2','PRZ0':'3','T':'1','F':'2','OC11':'1',
              'OC14':'2','OC12':'3','OC13':'4'}
X=np.copy(X_old)
for old, new in dictionary.items():
    X[X_old==old] = new

dictionary_y = {'T':'1','F':'0'}
y=np.copy(y_old)
for old, new in dictionary_y.items():
    y[y_old==old] = new

X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.2,random_state=0)
scaler=sklearn.preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

parameters = {
    'C': scipy.stats.reciprocal(0.1, 2.5),
    'solver': ["newton-cg", "lbfgs", "sag", "saga"],
    'multi_class': ["ovr", "multinomial"],
    'max_iter': [200]
    }
lrc=LogisticRegression(random_state=0, n_jobs=-1)
randcv = RandomizedSearchCV(lrc, parameters, n_iter=50, verbose=1, random_state=0, cv=5)
randcv.fit(X_train, y_train)

print(randcv.best_estimator_)
print(randcv.best_score_)
print(randcv.best_params_)

y_test_pred = randcv.best_estimator_.predict(X_test)
print(accuracy_score(y_test, y_test_pred))