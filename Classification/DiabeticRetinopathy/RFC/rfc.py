import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# read data (dataset at "https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set")
filename = "../../data/messidor_features.arff"
data = np.loadtxt(filename, delimiter=',', skiprows=24)
feature_removal_list = [[0],[1],np.arange(2,8),np.arange(8,16),[16],[17],[18]]

def applyRFC(features_to_be_removed):
    print('Features removed: ' + str(features_to_be_removed))
    X = data[:,:19]
    y = data[:,19]
    
    # preprocessing - scaling data and features removal
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.delete(X, np.s_[features_to_be_removed], axis=1)
    
    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    # cross validation for hyperparameter tuning
    n_estimators = [10,20,30,40,50,100]
    max_depth = [None,5,7,10,13,15,20]
    criterion = ["gini", "entropy"]
    parameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'criterion': criterion
            }
    rfc=RandomForestClassifier(random_state=0, n_jobs=-1)
    grid_search=GridSearchCV(rfc, param_grid=parameters, cv=10, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("\n\nBest Estimator: " + str(grid_search.best_estimator_))
    print("Score: " + str(grid_search.best_score_))
    print("Best params: " + str(grid_search.best_params_))
    
    # Test Data Accuracy Score
    y_test_pred = grid_search.best_estimator_.predict(X_test)
    print("Test Data Accuracy Score: " + str(accuracy_score(y_test, y_test_pred)))


for i,features_to_be_removed in enumerate(feature_removal_list):
    print('\n\n\n############################# Test ' + str(i) + ' #############################')
    applyRFC(features_to_be_removed)

print('\n\n\n############################# Model with all the features #############################')
applyRFC([])

print('\n\n\n############################# Final Test #############################')
final_features_to_be_removed = [0,16,17] 
print('Features observed to have negative effect on accuracy score: ' + str(final_features_to_be_removed))
applyRFC(final_features_to_be_removed)
    