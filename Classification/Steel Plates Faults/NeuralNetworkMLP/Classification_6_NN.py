# Dataset 6: Steel Plates Faults
# NN: Neural Network Classification
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
from sklearn.metrics import accuracy_score

data = np.loadtxt('Faults.NNA', delimiter='\t')
X=data[:,:27]
y=data[:,27:]
y=np.array([np.where(i==1)[0][0] for i in y])

X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.2,random_state=0)
scaler=sklearn.preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

parameters = {
    'hidden_layer_sizes': [(100,50,), (100,50,20,)],
    'learning_rate': ["constant", "invscaling", "adaptive"],
    'max_iter': [200, 250]
    }
mlp=MLPClassifier(random_state=0)
randcv = RandomizedSearchCV(mlp, parameters, n_iter=50, verbose=1, random_state=0, cv=5)
randcv.fit(X_train, y_train)

print(randcv.best_estimator_)
print(randcv.best_score_)
print(randcv.best_params_)

y_test_pred = randcv.best_estimator_.predict(X_test)
print(accuracy_score(y_test, y_test_pred))