import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# read data (dataset at "http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)")
filename = "../../data/australian.dat"
data = np.loadtxt(filename, delimiter=' ')
X = data[:,:14]
y = data[:,14]

# preprocessing - scaling data and features removal
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# cross validation for hyperparameter tuning
n_estimators = [10,20,30,40,50,100]
max_depth = [None,5,7,10,13,15,20]
criterion = ["gini", "entropy"]
parameters = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'criterion': criterion
        }
rfc=RandomForestClassifier(random_state=0, n_jobs=-1)
grid_search=GridSearchCV(rfc, param_grid=parameters, cv=10, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
    
print("\n\nBest Estimator: " + str(grid_search.best_estimator_))
print("Score: " + str(grid_search.best_score_))
print("Best params: " + str(grid_search.best_params_))

# Test Data Accuracy Score
y_test_pred = grid_search.best_estimator_.predict(X_test)
print("Test Data Accuracy Score: " + str(accuracy_score(y_test, y_test_pred)))