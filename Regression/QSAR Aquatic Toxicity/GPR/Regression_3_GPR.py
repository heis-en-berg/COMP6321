# Dataset 3: QSAR Aquatic Toxicity
# GPR: Gaussian Process Regression
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import DotProduct, RBF

data = np.loadtxt('qsar_aquatic_toxicity.csv', delimiter=';')
X=data[:,:8]
y=data[:,8:].reshape(-1,)
X=X.astype(np.float32)
y=y.astype(np.float32)

X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.2,random_state=0)
scaler=sklearn.preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

parameters = {
        #'kernel': [RBF(), RBF() + DotProduct()]
        'kernel': [RBF()]
        }
gpr=GaussianProcessRegressor(random_state=0)
randcv = RandomizedSearchCV(gpr, parameters, n_iter=50, verbose=1, random_state=0, cv=10)
randcv.fit(X_train, y_train)

print(randcv.best_estimator_)
print(randcv.best_score_)
print(randcv.best_params_)

y_test_pred = randcv.best_estimator_.predict(X_test)
print(mean_squared_error(y_test, y_test_pred))
