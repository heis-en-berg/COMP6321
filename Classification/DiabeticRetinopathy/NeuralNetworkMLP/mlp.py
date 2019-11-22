import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

# read data (dataset at "https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set")
filename = "../../data/messidor_features.arff"
data = np.loadtxt(filename, delimiter=',', skiprows=24)
feature_removal_list = [[0],[1],np.arange(2,8),np.arange(8,16),[16],[17],[18]]

def applyMLP(features_to_be_removed):
    print('Features removed: ' + str(features_to_be_removed))
    X = data[:,:19]
    y = data[:,19]
    
    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    # preprocessing - scaling data and features removal
    scaler_train = StandardScaler()
    X_train = scaler_train.fit_transform(X_train)
    scaler_test = StandardScaler()
    X_test = scaler_test.fit_transform(X_test)
    X_train = np.delete(X_train, np.s_[features_to_be_removed], axis=1)
    X_test = np.delete(X_test, np.s_[features_to_be_removed], axis=1)
    
    # cross validation for hyperparameter tuning
    param_distributions = {
            'hidden_layer_sizes': [(100,50,), (100,50,20,)],
            'learning_rate': ["constant", "invscaling", "adaptive"],
            'max_iter': [200, 250]
            }
    mlp=MLPClassifier(random_state=0)
    randcv = RandomizedSearchCV(mlp, param_distributions, n_iter=50, verbose=1, random_state=0, cv=5, n_jobs=-1)
    randcv.fit(X_train, y_train)
    
    print("\n\nBest Estimator: " + str(randcv.best_estimator_))
    print("Score: " + str(randcv.best_score_))
    print("Best params: " + str(randcv.best_params_))
    
    # Test Data Accuracy Score
    y_test_pred = randcv.best_estimator_.predict(X_test)
    print("Test Data Accuracy Score: " + str(accuracy_score(y_test, y_test_pred)))


for i,features_to_be_removed in enumerate(feature_removal_list):
    print('\n\n\n############################# Test ' + str(i) + ' #############################')
    applyMLP(features_to_be_removed)

print('\n\n\n############################# Model with all the features #############################')
applyMLP([])

print('\n\n\n############################# Final Test #############################')
final_features_to_be_removed = [17] 
print('Features observed to have negative effect on accuracy score: ' + str(final_features_to_be_removed))
applyMLP(final_features_to_be_removed)
    