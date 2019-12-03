# Dataset 4: Parkinson Speech
# ABR: AdaBoost Regression
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
from sklearn.metrics import mean_squared_error

data = np.loadtxt('train_data.txt', delimiter=',')
X=data[:,1:27]
y=data[:,28:].reshape(-1,)
X=X.astype(np.float32)
y=y.astype(np.float32)

data = np.loadtxt('test_data.txt', delimiter=',')
X_testFromFile=data[:,1:27]
y_testFromFile=data[:,27:].reshape(-1,)
X_testFromFile=X_testFromFile.astype(np.float32)
y_testFromFile=y_testFromFile.astype(np.float32)

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
print("From Training Dataset: ",mean_squared_error(y_test, y_test_pred))

y_testFromFile_pred = randcv.best_estimator_.predict(X_testFromFile)
print("From Test Dataset: ",mean_squared_error(y_testFromFile, y_testFromFile_pred))
