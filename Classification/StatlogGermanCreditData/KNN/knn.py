import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

'''
    TODO::not sure which features to use from data
'''
# read data (dataset at "https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)")
filename = "../../data/german.data-numeric"
data = np.genfromtxt(filename, autostrip=True)
X = data[:,:20]
y = data[:,20]

# preprocessing - scaling data and features removal
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# cross validation for hyperparameter tuning
n_neighbors = [1,3,5,7,9,11,15,17,19]
algorithm = ["ball_tree", "kd_tree"]
metric = ["euclidean", "manhattan", "chebyshev", "minkowski"]
parameters = {
    'n_neighbors': n_neighbors,
    'metric': metric,
    'algorithm': algorithm
    }
knn=KNeighborsClassifier()
grid_search=GridSearchCV(knn, param_grid=parameters, cv=10, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\n\nBest Estimator: " + str(grid_search.best_estimator_))
print("Score: " + str(grid_search.best_score_))
print("Best params: " + str(grid_search.best_params_))

# Test Data Accuracy Score
y_test_pred = grid_search.best_estimator_.predict(X_test)
print("Test Data Accuracy Score: " + str(accuracy_score(y_test, y_test_pred)))

print(confusion_matrix(y, grid_search.best_estimator_.predict(X)))
