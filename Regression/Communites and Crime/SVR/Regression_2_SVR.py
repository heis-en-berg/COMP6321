# Dataset 2: Communities and Crime
# SVR: Support Vector Regression
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
from sklearn.metrics import mean_squared_error

data = np.loadtxt('communities.data', delimiter=',',dtype='str')
X_old=np.concatenate((data[:,5:101],data[:,118:121]),axis=1)
X_old=np.concatenate((X_old,data[:,125:126]),axis=1)
y=data[:,127:].reshape(-1,)

letters = {'?':'0'}
X=np.copy(X_old)
for old, new in letters.items():
    X[X_old==old] = new

X=X.astype(np.float32)
y=y.astype(np.float32)
X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.2,random_state=0)
scaler=sklearn.preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

parameters = {
        'degree': scipy.stats.randint(1,10),
        'kernel': ['linear', 'poly', 'rbf'],
        'gamma': scipy.stats.randint(1,10),
        'coef0': scipy.stats.randint(1,5)
        }
svr=SVR()
randcv = RandomizedSearchCV(svr, parameters, n_iter=100, verbose=1, random_state=0, cv=10)
randcv.fit(X_train, y_train)

print(randcv.best_estimator_)
print(randcv.best_score_)
print(randcv.best_params_)

y_test_pred = randcv.best_estimator_.predict(X_test)
print(mean_squared_error(y_test, y_test_pred))