# Dataset 5: Facebook Metrics
# NNR: Neural Network Regression
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
from sklearn.metrics import mean_squared_error

data = np.loadtxt('dataset_Facebook.csv', delimiter=';', skiprows=1, dtype='str')
X_old=data[:,:7]
#From the 12 output we chose 'Lifetime Post Consumer'. We must cite Moro's paper
y_11_old=data[:,10:11].reshape(-1,)


dictionary = {'Photo':'1','Status':'2','Link':'3','Video':'4','':'0'}
X=np.copy(X_old)
for old, new in dictionary.items():
    X[X_old==old] = new

y=np.copy(y_11_old)
for old, new in dictionary.items():
    y[y_11_old==old] = new

X=X.astype(np.int32)
y=y.astype(np.int32)

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
