from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle

def train_and_save_final_model(X, y, X_train, y_train, params, save_model_file_path, test_data):
    rfr=RandomForestRegressor(random_state=0)
    rfr.set_params(**params)
    
    if test_data == None:
        rfr.fit(X_train, y_train)
    else:
        rfr.fit(X, y)
    
    #save model
    model_file_path = save_model_file_path + 'rfr.sav'
    pickle.dump(rfr, open(model_file_path, 'wb'))
    

def fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path, test_data):
    # cross validation for hyperparameter tuning
    param_distributions = {
            'n_estimators': [10,50,100,150],
            'max_depth': [10,20,30,50]
            }
    rfr=RandomForestRegressor(random_state=0)
    randcv = RandomizedSearchCV(rfr, param_distributions, random_state=0, cv=5)
    randcv.fit(X_train, y_train)
    
    # final training
    train_and_save_final_model(X, y, X_train, y_train, randcv.best_params_, save_model_file_path, test_data)