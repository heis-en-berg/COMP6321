from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
import scipy
import pickle

def train_and_save_final_model(X, y, X_train, y_train, params, save_model_file_path, test_data):
    knn=KNeighborsClassifier()
    knn.set_params(**params)
    
    if test_data == None:
        knn.fit(X_train, y_train)
    else:
        knn.fit(X, y)
    
    #save model
    model_file_path = save_model_file_path + 'knn.sav'
    pickle.dump(knn, open(model_file_path, 'wb'))

def fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path, test_data):
    # cross validation for hyperparameter tuning    
    param_distributions = {
            'n_neighbors' : scipy.stats.randint(1,30),
            'metric' : ["euclidean", "manhattan", "chebyshev", "minkowski"]
    }
    
    knn=KNeighborsClassifier()
    randcv = RandomizedSearchCV(knn, param_distributions, random_state=0, cv=5)
    randcv.fit(X_train, y_train)
    
    # final training
    train_and_save_final_model(X, y, X_train, y_train, randcv.best_params_, save_model_file_path, test_data)

