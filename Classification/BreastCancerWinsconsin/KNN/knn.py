import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# read data (dataset at "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)")
filename = "../data/wdbc.data"
feature_columns_index = np.arange(2,32)
label_column_index = 1
X = np.loadtxt(filename, delimiter=',', usecols = feature_columns_index)
y = np.loadtxt(filename, delimiter=',', usecols = label_column_index, dtype=np.str)

# preprocessing
unique_labels = np.unique(y)
le = preprocessing.LabelEncoder()
le.fit(unique_labels)
y = le.transform(y)

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# cross validation for hyperparameter tuning
n_neighbors = [1,3,5,7,9,11,13,15]
algorithm = ["ball_tree", "kd_tree", "brute", "auto"]
metric = ["euclidean", "manhattan", "chebyshev", "minkowski"]
algorithm
parameters = {
        'n_neighbors': n_neighbors,
        'metric': metric,
        'algorithm': algorithm
        }
knn=KNeighborsClassifier()
grid_search=GridSearchCV(knn, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\n\nBest Estimator: " + str(grid_search.best_estimator_))
print("Score: " + str(grid_search.best_score_))
print("Best params: " + str(grid_search.best_params_))

# Test Data Accuracy Score
y_test_pred = grid_search.best_estimator_.predict(X_test)
print("Test Data Accuracy Score: " + str(accuracy_score(y_test, y_test_pred)))

#print(grid_search.cv_results_.keys())