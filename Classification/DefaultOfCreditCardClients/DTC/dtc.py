import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# read data (dataset at "https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients")
filename = "../../data/data.xls"
data = np.asarray(pd.read_excel(filename, index_col=0, skiprows=1))

feature_removal_list = [[0],[1],[2],[3],[4],np.arange(5,11),np.arange(11,17),np.arange(17,23)]

def applyDTC(features_to_be_removed):
    print('Features removed: ' + str(features_to_be_removed))
    X = data[:,:23]
    y = data[:,23]
    
    # preprocessing - scaling data and features removal
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.delete(X, np.s_[features_to_be_removed], axis=1)
    
    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    # cross validation for hyperparameter tuning
    max_depth = [None,5,7,10,13,15,20]
    criterion = ["gini", "entropy"]
    parameters = {
            'max_depth': max_depth,
            'criterion': criterion
            }
    dtc=DecisionTreeClassifier(random_state=0)
    grid_search=GridSearchCV(dtc, param_grid=parameters, cv=10, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("\n\nBest Estimator: " + str(grid_search.best_estimator_))
    print("Score: " + str(grid_search.best_score_))
    print("Best params: " + str(grid_search.best_params_))
    
    # Test Data Accuracy Score
    y_test_pred = grid_search.best_estimator_.predict(X_test)
    print("Test Data Accuracy Score: " + str(accuracy_score(y_test, y_test_pred)))


for i,features_to_be_removed in enumerate(feature_removal_list):
    print('\n\n\n############################# Test ' + str(i) + ' #############################')
    applyDTC(features_to_be_removed)

print('\n\n\n############################# Model with all the features #############################')
applyDTC([])

print('\n\n\n############################# Final Test #############################')
# It was observed that features from 1 to 10 and 11 to 22 didn't matter to the 
#final model because they didn't contribute much to the training
final_features_to_be_removed = np.concatenate((np.arange(0,5),np.arange(11,23)))
print('Features observed to have negative effect on accuracy score: ' + str(final_features_to_be_removed))
applyDTC(final_features_to_be_removed)
    