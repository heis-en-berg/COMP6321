from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
import scipy
import pickle

def train_and_save_final_model(X, y, params, save_model_file_path):
    dtr=DecisionTreeRegressor(random_state=0)
    dtr.set_params(**params)
    dtr.fit(X, y)
    
    #save model
    model_file_path = save_model_file_path + 'dtr.sav'
    pickle.dump(dtr, open(model_file_path, 'wb'))
    

def fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path):
    # cross validation for hyperparameter tuning
    param_distributions = {
            'criterion': ['mse', 'mae'],
            'max_depth': scipy.stats.randint(10,50)
            }
    dtr=DecisionTreeRegressor(random_state=0)
    randcv = RandomizedSearchCV(dtr, param_distributions, random_state=0, cv=5)
    randcv.fit(X_train, y_train)
    
    # final training
    train_and_save_final_model(X, y, randcv.best_params_, save_model_file_path)