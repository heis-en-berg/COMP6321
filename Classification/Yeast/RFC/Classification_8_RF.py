# Dataset 8: Yeast
# RF: Random Forest Classification
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
from sklearn.metrics import accuracy_score

data = np.loadtxt('yeast.data',dtype='str')
X=data[:,1:9]
y_old=data[:,9:]

dictionary_y = {'MIT':'1','NUC':'2','CYT':'3','ME1':'4','ME2':'4',
                'ME3':'5','EXC':'6','POX':'7'}
y=np.copy(y_old)
for old, new in dictionary_y.items():
    y[y_old==old] = new

X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.2,random_state=0)
scaler=sklearn.preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

parameters = {
    'n_estimators': np.linspace(10,100,10, dtype=np.int32),
    'max_depth': np.linspace(1,30,10, dtype=np.int32),
    'criterion': ["gini", "entropy"]
    }
rfc=RandomForestClassifier(random_state=0, n_jobs=-1)
randcv = RandomizedSearchCV(rfc, parameters, n_iter=100, verbose=1, random_state=0, cv=5)
randcv.fit(X_train, y_train)

print(randcv.best_estimator_)
print(randcv.best_score_)
print(randcv.best_params_)

y_test_pred = randcv.best_estimator_.predict(X_test)
print(accuracy_score(y_test, y_test_pred))