# Dataset 6: Steel Plates Faults
# KNN: K-Nearest Neighbours Classification
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.neighbors import KNeighborsClassifier
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
    'n_neighbors' : np.linspace(1,30,10, dtype=np.int32),
    'algorithm' : ["ball_tree", "kd_tree", "brute", "auto"],
    'metric' : ["euclidean", "manhattan", "chebyshev", "minkowski"]
}

knn=KNeighborsClassifier()
randcv = RandomizedSearchCV(knn, parameters, n_iter=50, verbose=1, random_state=0, cv=5)
randcv.fit(X_train, y_train)

print(randcv.best_estimator_)
print(randcv.best_score_)
print(randcv.best_params_)

y_test_pred = randcv.best_estimator_.predict(X_test)
print(accuracy_score(y_test, y_test_pred))
