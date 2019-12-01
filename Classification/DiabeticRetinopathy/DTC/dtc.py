import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
import scipy
from sklearn.metrics import accuracy_score

# read data (dataset at "https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set")
filename = "../../data/messidor_features.arff"
data = np.loadtxt(filename, delimiter=',', skiprows=24)
feature_removal_list = [[0],[1],np.arange(2,8),np.arange(8,16),[16],[17],[18]]

def applyDTC(features_to_be_removed):
    print('Features removed: ' + str(features_to_be_removed))
    X = data[:,:19]
    y = data[:,19]
    
    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    # preprocessing - scaling data and features removal
    X_train = np.delete(X_train, np.s_[features_to_be_removed], axis=1)
    X_test = np.delete(X_test, np.s_[features_to_be_removed], axis=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    # cross validation for hyperparameter tuning
    param_distributions = {
            'max_depth': scipy.stats.randint(1,30),
            'criterion': ["gini", "entropy"]
            }
    dtc=DecisionTreeClassifier(random_state=0)
    randcv = RandomizedSearchCV(dtc, param_distributions, n_iter=50, verbose=1, random_state=0, cv=5)
    randcv.fit(X_train, y_train)
    
    print("\n\nBest Estimator: " + str(randcv.best_estimator_))
    print("Score: " + str(randcv.best_score_))
    print("Best params: " + str(randcv.best_params_))
    
    # Test Data Accuracy Score
    y_test_pred = randcv.best_estimator_.predict(X_test)
    print("Test Data Accuracy Score: " + str(accuracy_score(y_test, y_test_pred)))


for i,features_to_be_removed in enumerate(feature_removal_list):
    print('\n\n\n############################# Test ' + str(i) + ' #############################')
    applyDTC(features_to_be_removed)

print('\n\n\n############################# Model with all the features #############################')
applyDTC([])

print('\n\n\n############################# Final Test #############################')
final_features_to_be_removed = [0,16,17] 
print('Features observed to have negative effect on accuracy score: ' + str(final_features_to_be_removed))
applyDTC(final_features_to_be_removed)
    