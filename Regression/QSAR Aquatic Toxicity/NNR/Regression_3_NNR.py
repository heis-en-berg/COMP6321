# Dataset 3: QSAR Aquatic Toxicity
# NNR: Neural Network Regression
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
from sklearn.metrics import mean_squared_error

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
        'hidden_layer_sizes': [(100,50,), (100,50,20,)],
        'activation':['tanh','relu'],
        'solver':['sgd','adam'],
        'alpha':np.logspace(-2, 1, num=4),
        'learning_rate': ["constant", "invscaling", "adaptive"],
        'max_iter': [200, 250, 300]
        }
mlpr=MLPRegressor(random_state=0)
randcv = RandomizedSearchCV(mlpr, parameters, n_iter=500, verbose=1, random_state=0, cv=10)
randcv.fit(X_train, y_train)

print(randcv.best_estimator_)
print(randcv.best_score_)
print(randcv.best_params_)

y_test_pred = randcv.best_estimator_.predict(X_test)
print(mean_squared_error(y_test, y_test_pred))