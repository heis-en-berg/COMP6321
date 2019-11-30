import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
import scipy
from matplotlib.dates import datestr2num
from sklearn.metrics import mean_squared_error

# read data (dataset at "http://archive.ics.uci.edu/ml/datasets/Student+Performance")
def convert_date(date_bytes):
    return datestr2num(date_bytes.decode('ascii'))

filename = "../../data/student/student-por.csv"
data = pd.read_csv(filename, delimiter=';')
columns = data.columns
X = data.loc[:,columns[:32]]
y = data.loc[:,columns[32]]

# labeling string columns

def label_column(columns):
    global X
    for column in columns:
        X=pd.concat([X,pd.get_dummies(X[column],prefix=column)],axis=1).drop([column],axis=1)

columns_to_be_labeled = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                         'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                         'famsup', 'paid', 'activities', 'nursery', 'higher',
                         'internet', 'romantic']
label_column(columns_to_be_labeled)

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# cross validation for hyperparameter tuning
param_distributions = {
        'n_estimators': scipy.stats.randint(50,200),
        'learning_rate': scipy.stats.reciprocal(0.1,2.0),
        'loss': ['linear', 'square', 'exponential']
        }
adbr=AdaBoostRegressor(random_state=0)
randcv = RandomizedSearchCV(adbr, param_distributions, n_iter=100, verbose=1, random_state=0, cv=10)
randcv.fit(X_train, y_train)

print("\n\nBest Estimator: " + str(randcv.best_estimator_))
print("Score: " + str(randcv.best_score_))
print("Best params: " + str(randcv.best_params_))

# Test Data Accuracy Score
y_test_pred = randcv.best_estimator_.predict(X_test)
print("Test Data mean_squared_error: " + str(mean_squared_error(y_test, y_test_pred)))