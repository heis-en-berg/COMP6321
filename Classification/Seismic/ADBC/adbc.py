import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import scipy.stats

# read data (dataset at "http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)")
filename = "../../data/seismic-bumps.arff"

categorcal_features = [0,1,2,7]
non_categorical_features = np.concatenate((np.arange(3,7),np.arange(8,17)))
class_column = 18
X_categorical_features = np.loadtxt(filename, delimiter=',', skiprows=154, dtype=np.str, usecols=categorcal_features)
X_non_categorical_features = np.loadtxt(filename, delimiter=',', skiprows=154, dtype=np.int32, usecols=non_categorical_features)
y = np.loadtxt(filename, delimiter=',', skiprows=154, dtype=np.int32, usecols=class_column)

# Encoding categorical features

def label_encode_column(column):
    global X_categorical_features
    unique_labels = np.unique(X_categorical_features[:,column])
    le = LabelEncoder()
    le.fit(unique_labels)
    X_categorical_features[:,column] = le.transform(X_categorical_features[:,column])

for column in np.arange(X_categorical_features.shape[1]):
    label_encode_column(column)
    
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(X_categorical_features)
X_categorical_features = one_hot_encoder.transform(X_categorical_features).toarray()

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# preprocessing - scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# cross validation for hyperparameter tuning
param_distributions = {
        'n_estimators': scipy.stats.randint(50,200),
        'learning_rate': scipy.stats.reciprocal(1.0, 3.0),
        'algorithm': ["SAMME", "SAMME.R"]
}
adbc=AdaBoostClassifier(random_state=0)
randcv = RandomizedSearchCV(adbc, param_distributions, n_iter=100, verbose=1, random_state=0, cv=10)
randcv.fit(X_train, y_train)

print("\n\nBest Estimator: " + str(randcv.best_estimator_))
print("Score: " + str(randcv.best_score_))
print("Best params: " + str(randcv.best_params_))

# Test Data Accuracy Score
y_test_pred = randcv.best_estimator_.predict(X_test)
print("Test Data Accuracy Score: " + str(accuracy_score(y_test, y_test_pred)))