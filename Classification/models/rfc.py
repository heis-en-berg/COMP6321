from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import scipy
import pickle

def train_and_save_final_model(X, y, X_train, y_train, params, save_model_file_path, test_data):
    rfc=RandomForestClassifier(random_state=0, n_jobs=-1)
    rfc.set_params(**params)
    
    if test_data == None:
        rfc.fit(X_train, y_train)
    else:
        rfc.fit(X, y)
    
    #save model
    model_file_path = save_model_file_path + 'rfc.sav'
    pickle.dump(rfc, open(model_file_path, 'wb'))
    

def fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path, test_data):
    # cross validation for hyperparameter tuning
    param_distributions = {
            'n_estimators': scipy.stats.randint(10,100),
            'max_depth': scipy.stats.randint(1,30),
            }
    rfc=RandomForestClassifier(random_state=0, n_jobs=-1)
    randcv = RandomizedSearchCV(rfc, param_distributions, random_state=0, cv=5)
    randcv.fit(X_train, y_train)
    
    # final training
    train_and_save_final_model(X, y, X_train, y_train, randcv.best_params_, save_model_file_path, test_data)
