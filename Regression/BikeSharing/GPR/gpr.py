import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.gaussian_process.kernels import DotProduct, RBF
from matplotlib.dates import datestr2num
from sklearn.metrics import mean_squared_error

# read data (dataset at "http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset")
def convert_date(date_bytes):
    return datestr2num(date_bytes.decode('ascii'))

filename = "../../data/Bike-Sharing-Dataset/hour.csv"
data = np.loadtxt(filename, delimiter=',', skiprows=1, converters = {1: convert_date})
X = data[:,:16]
y = data[:,16]

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# scaling data
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.fit_transform(X_test)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.values.reshape(-1,1)).reshape(-1)
y_test = y_scaler.fit_transform(y_test.values.reshape(-1,1)).reshape(-1)

# cross validation for hyperparameter tuning
param_distributions = {
        'kernel': [RBF(), RBF() + DotProduct()]
        }
gpr=GaussianProcessRegressor(random_state=0)
randcv = RandomizedSearchCV(gpr, param_distributions, n_iter=100, verbose=1, random_state=0, cv=10)
randcv.fit(X_train, y_train)

print("\n\nBest Estimator: " + str(randcv.best_estimator_))
print("Score: " + str(randcv.best_score_))
print("Best params: " + str(randcv.best_params_))

# Test Data Accuracy Score
y_test_pred = randcv.best_estimator_.predict(X_test)
print("Test Data mean_squared_error: " + str(mean_squared_error(y_test, y_test_pred)))