import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
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

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# preprocessing - scaling data and features removal
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# cross validation for hyperparameter tuning
param_distributions = {
        'hidden_layer_sizes': [(100,50,), (100,50,20,)],
        'learning_rate': ["constant", "invscaling", "adaptive"],
        'max_iter': [200, 250]
        }
mlp=MLPClassifier(random_state=0)
randcv = RandomizedSearchCV(mlp, param_distributions, n_iter=50, verbose=1, random_state=0, cv=10)
randcv.fit(X_train, y_train)

print("\n\nBest Estimator: " + str(randcv.best_estimator_))
print("Score: " + str(randcv.best_score_))
print("Best params: " + str(randcv.best_params_))

# Test Data Accuracy Score
y_test_pred = randcv.best_estimator_.predict(X_test)
print("Test Data Accuracy Score: " + str(accuracy_score(y_test, y_test_pred)))
