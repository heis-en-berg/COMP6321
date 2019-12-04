from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
import pickle

def train_and_save_final_model(X, y, params, save_model_file_path):
    svr=SVR()
    svr.set_params(**params)
    svr.fit(X, y)
    
    #save model
    model_file_path = save_model_file_path + 'svr.sav'
    pickle.dump(svr, open(model_file_path, 'wb'))
    

def fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path):
    # cross validation for hyperparameter tuning
    param_distributions = {
            'C'     : scipy.stats.reciprocal(1.0, 100.0),
            'gamma': scipy.stats.randint(1,10),
            'max_iter' : [20000],
            'cache_size' : [1000]
            }
    svr=SVR()
    randcv = RandomizedSearchCV(svr, param_distributions, random_state=0, cv=5)
    randcv.fit(X_train, y_train)
    
    # final training
    train_and_save_final_model(X, y, randcv.best_params_, save_model_file_path)