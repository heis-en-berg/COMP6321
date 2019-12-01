import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

# read data (dataset at "https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients")
filename = "../../data/data.xls"
data = np.asarray(pd.read_excel(filename, index_col=0, skiprows=1))

feature_removal_list = [[0],[1],[2],[3],[4],np.arange(5,11),np.arange(11,17),np.arange(17,23)]

def applyGNBC(features_to_be_removed):
    print('Features removed: ' + str(features_to_be_removed))
    X = data[:,:23]
    y = data[:,23]
    
     # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    # preprocessing - scaling data and features removal
    X_train = np.delete(X_train, np.s_[features_to_be_removed], axis=1)
    X_test = np.delete(X_test, np.s_[features_to_be_removed], axis=1)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    # cross validation for hyperparameter tuning
    param_distributions = {}
    gnbc=GaussianNB()
    randcv = RandomizedSearchCV(gnbc, param_distributions, n_iter=100, verbose=1, random_state=0, cv=5)
    randcv.fit(X_train, y_train)
    
    print("\n\nBest Estimator: " + str(randcv.best_estimator_))
    print("Score: " + str(randcv.best_score_))
    print("Best params: " + str(randcv.best_params_))
    
    # Test Data Accuracy Score
    y_test_pred = randcv.best_estimator_.predict(X_test)
    print("Test Data Accuracy Score: " + str(accuracy_score(y_test, y_test_pred)))


for i,features_to_be_removed in enumerate(feature_removal_list):
    print('\n\n\n############################# Test ' + str(i) + ' #############################')
    applyGNBC(features_to_be_removed)

print('\n\n\n############################# Model with all the features #############################')
applyGNBC([])

print('\n\n\n############################# Final Test #############################')
# It was observed that features from 1 to 10 and 11 to 22 didn't matter to the 
#final model because they didn't contribute much to the training
final_features_to_be_removed = np.concatenate((np.arange(0,4),np.arange(11,23)))
print('Features observed to have negative effect on accuracy score: ' + str(final_features_to_be_removed))
applyGNBC(final_features_to_be_removed)
    