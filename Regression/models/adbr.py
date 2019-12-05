from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
import pickle

def train_and_save_final_model(X, y, X_train, y_train, params, save_model_file_path, test_data):
    adbr=AdaBoostRegressor(random_state=0)
    adbr.set_params(**params)

    if test_data == None:
        adbr.fit(X_train, y_train)
    else:
        adbr.fit(X, y)
    
    #save model
    model_file_path = save_model_file_path + 'adbr.sav'
    pickle.dump(adbr, open(model_file_path, 'wb'))

def fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path, test_data):
    # cross validation for hyperparameter tuning
    param_distributions = {
            'n_estimators': scipy.stats.randint(50,200),
            'learning_rate': scipy.stats.reciprocal(0.1,2.0),
            }
    adbr=AdaBoostRegressor(random_state=0)
    randcv = RandomizedSearchCV(adbr, param_distributions, random_state=0, cv=5)
    randcv.fit(X_train, y_train)
    
    # final training
    train_and_save_final_model(X, y, X_train, y_train, randcv.best_params_, save_model_file_path, test_data)
