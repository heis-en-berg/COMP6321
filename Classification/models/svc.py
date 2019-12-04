from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import scipy.stats
import pickle

def train_and_save_final_model(X, y, params, save_model_file_path):
    svc=SVC(random_state=0)
    svc.set_params(**params)
    svc.fit(X, y)
    
    #save model
    model_file_path = save_model_file_path + 'svc.sav'
    pickle.dump(svc, open(model_file_path, 'wb'))
    

def fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path):
    # cross validation for hyperparameter tuning    
    param_distributions = {
            'C'     : scipy.stats.reciprocal(1.0, 100.0),
            'gamma' : scipy.stats.randint(1,15),
            'max_iter' : [20000],
            'cache_size' : [1000]
    }
    
    svc=SVC(random_state=0)
    randcv = RandomizedSearchCV(svc, param_distributions, n_iter=50, random_state=0, cv=5)
    randcv.fit(X_train, y_train)
    
    # final training
    train_and_save_final_model(X, y, randcv.best_params_, save_model_file_path)
    
    # Test Data Accuracy Score
    y_test_pred = randcv.best_estimator_.predict(X_test)
    best_params = randcv.best_params_
    test_data_accuracy_score = accuracy_score(y_test, y_test_pred)
    return best_params, test_data_accuracy_score