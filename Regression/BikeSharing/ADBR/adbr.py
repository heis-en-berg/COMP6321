import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
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

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# cross validation for hyperparameter tuning
n_estimators = [10,50,100]
learning_rate = [0.1,1.0,1.5]
loss = ['linear', 'square', 'exponential']
parameters = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'loss': loss
        }
adbr=AdaBoostRegressor(random_state=0)
grid_search=GridSearchCV(adbr, param_grid=parameters, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\n\nBest Estimator: " + str(grid_search.best_estimator_))
print("Score: " + str(grid_search.best_score_))
print("Best params: " + str(grid_search.best_params_))

# Test Data Accuracy Score
y_test_pred = grid_search.best_estimator_.predict(X_test)
print("Test Data mean_squared_error: " + str(mean_squared_error(y_test, y_test_pred)))