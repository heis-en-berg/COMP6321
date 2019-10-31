import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

'''
NOTE: >3 minutes
'''

# read data (dataset at "https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set")
filename = "../../data/messidor_features.arff"
data = np.loadtxt(filename, delimiter=',', skiprows=24)
feature_removal_list = [[0],[1],np.arange(2,8),np.arange(8,16),[16],[17],[18]]

def applySVC(features_to_be_removed):
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
    kernel = ["poly"]#, "rbf"]
    degree = [5]#np.linspace(1,5,5)
    gamma = [5]#np.linspace(1,5,5)
    coef0 = [0]#[0,0.1,1,2,3]
    parameters = {
            'kernel': kernel,
            'degree': degree,
            'gamma': gamma,
            'coef0': coef0
            }
    svc=SVC(random_state=0)
    grid_search=GridSearchCV(svc, param_grid=parameters, cv=2, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("\n\nBest Estimator: " + str(grid_search.best_estimator_))
    print("Score: " + str(grid_search.best_score_))
    print("Best params: " + str(grid_search.best_params_))
    
    # Test Data Accuracy Score
    y_test_pred = grid_search.best_estimator_.predict(X_test)
    print("Test Data Accuracy Score: " + str(accuracy_score(y_test, y_test_pred)))


for i,features_to_be_removed in enumerate(feature_removal_list):
    print('\n\n\n############################# Test ' + str(i) + ' #############################')
    applySVC(features_to_be_removed)

print('\n\n\n############################# Model with all the features #############################')
applySVC([])

print('\n\n\n############################# Final Test #############################')
final_features_to_be_removed = [0,16,17] 
print('Features observed to have negative effect on accuracy score: ' + str(final_features_to_be_removed))
applySVC(final_features_to_be_removed)