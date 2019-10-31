import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# read data (dataset at "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)")
filename = "../../data/wdbc.data"
feature_columns_index = np.arange(2,32)
label_column_index = 1
X = np.loadtxt(filename, delimiter=',', usecols = feature_columns_index)
y = np.loadtxt(filename, delimiter=',', usecols = label_column_index, dtype=np.str)

# preprocessing
unique_labels = np.unique(y)
le = preprocessing.LabelEncoder()
le.fit(unique_labels)
y = le.transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# cross validation for hyperparameter tuning
kernel = ["poly", "rbf"]
degree = np.linspace(1,5,5)
gamma = np.linspace(1,5,5)
coef0 = [0,0.1,1,2,3]
parameters = {
        'kernel': kernel,
        'degree': degree,
        'gamma': gamma,
        'coef0': coef0
        }
svc=SVC(random_state=0)
grid_search=GridSearchCV(svc, param_grid=parameters, cv=10, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\n\nBest Estimator: " + str(grid_search.best_estimator_))
print("Score: " + str(grid_search.best_score_))
print("Best params: " + str(grid_search.best_params_))

# Test Data Accuracy Score
y_test_pred = grid_search.best_estimator_.predict(X_test)
print("Test Data Accuracy Score: " + str(accuracy_score(y_test, y_test_pred)))