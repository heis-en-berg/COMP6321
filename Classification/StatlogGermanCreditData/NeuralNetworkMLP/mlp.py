import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

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

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# preprocessing - scaling data and features removal
scaler_train = StandardScaler()
X_train = scaler_train.fit_transform(X_train)
scaler_test = StandardScaler()
X_test = scaler_test.fit_transform(X_test)

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