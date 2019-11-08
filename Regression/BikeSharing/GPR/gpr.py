import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, ExpSineSquared, RBF
from sklearn.model_selection import GridSearchCV
from matplotlib.dates import datestr2num
from sklearn.metrics import mean_squared_error

# read data (dataset at "http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset")
def convert_date(date_bytes):
    return datestr2num(date_bytes.decode('ascii'))

filename = "../../data/Bike-Sharing-Dataset/hour.csv"
data = np.loadtxt(filename, delimiter=',', skiprows=1, converters = {1: convert_date})
X = data[:,:16]
y = data[:,16]

# preprocessing - scaling data and features removal
scaler = StandardScaler()
X = scaler.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1,1)).reshape(-1)

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# cross validation for hyperparameter tuning
kernel = [RBF()]#, ExpSineSquared(), RBF(), RBF() + DotProduct()]
parameters = {
        'kernel': kernel
        }
gpr=GaussianProcessRegressor(random_state=0, kernel=RBF())
#grid_search=GridSearchCV(gpr, param_grid=parameters, cv=5, verbose=1, n_jobs=-1)
gpr.fit(X_train, y_train)

#print("\n\nBest Estimator: " + str(grid_search.best_estimator_))
print("Score: " + str(gpr.score(X_train, y_train)))
#print("Best params: " + str(grid_search.best_params_))

# Test Data Accuracy Score
y_test_pred = gpr.predict(X_test)
print("Test Data mean_squared_error: " + str(mean_squared_error(y_test, y_test_pred)))