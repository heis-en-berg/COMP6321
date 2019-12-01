import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# read data (dataset at "http://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance")
filename = "../../data/sgemm_product_dataset/sgemm_product.csv"
data = np.loadtxt(filename, delimiter=',', skiprows=1)
X = data[:,:14]
y = data[:,14:18]

# averaging values row wise in y
y = y.mean(axis=1)

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# scaling data
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.fit_transform(X_test)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).reshape(-1)
y_test = y_scaler.fit_transform(y_test.reshape(-1,1)).reshape(-1)

# cross validation for hyperparameter tuning
param_distributions = {
        'hidden_layer_sizes': [(100,50,), (100,50,20,)],
        'learning_rate': ["constant", "invscaling", "adaptive"],
        'max_iter': [200, 250, 300]
        }
mlpr=MLPRegressor(random_state=0)
randcv = RandomizedSearchCV(mlpr, param_distributions, n_iter=100, verbose=1, random_state=0, cv=10)
randcv.fit(X_train, y_train)

print("\n\nBest Estimator: " + str(randcv.best_estimator_))
print("Score: " + str(randcv.best_score_))
print("Best params: " + str(randcv.best_params_))

# Test Data Accuracy Score
y_test_pred = randcv.best_estimator_.predict(X_test)
print("Test Data mean_squared_error: " + str(mean_squared_error(y_test, y_test_pred)))