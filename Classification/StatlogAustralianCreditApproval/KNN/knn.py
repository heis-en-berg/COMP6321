import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
import scipy
from sklearn.metrics import accuracy_score

# read data (dataset at "http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)")
filename = "../../data/australian.dat"
data = np.loadtxt(filename, delimiter=' ')
X = data[:,:14]
y = data[:,14]

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# preprocessing - scaling data and features removal
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# cross validation for hyperparameter tuning    
param_distributions = {
        'n_neighbors' : scipy.stats.randint(1,30),
        'algorithm' : ["ball_tree", "kd_tree", "brute", "auto"],
        'metric' : ["euclidean", "manhattan", "chebyshev", "minkowski"]
}

knn=KNeighborsClassifier()
randcv = RandomizedSearchCV(knn, param_distributions, n_iter=50, verbose=1, random_state=0, cv=10)
randcv.fit(X_train, y_train)

print("\n\nBest Estimator: " + str(randcv.best_estimator_))
print("Score: " + str(randcv.best_score_))
print("Best params: " + str(randcv.best_params_))

# Test Data Accuracy Score
y_test_pred = randcv.best_estimator_.predict(X_test)
print("Test Data Accuracy Score: " + str(accuracy_score(y_test, y_test_pred)))