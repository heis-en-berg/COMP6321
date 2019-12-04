from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import scipy
import pickle

def train_and_save_final_model(X, y, params, save_model_file_path):
    rfc=RandomForestClassifier(random_state=0, n_jobs=-1)
    rfc.set_params(**params)
    rfc.fit(X, y)
    
    #save model
    model_file_path = save_model_file_path + 'rfc.sav'
    pickle.dump(rfc, open(model_file_path, 'wb'))
    

def fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path):
    # cross validation for hyperparameter tuning
    param_distributions = {
            'n_estimators': scipy.stats.randint(10,100),
            'max_depth': scipy.stats.randint(1,30),
            }
    rfc=RandomForestClassifier(random_state=0, n_jobs=-1)
    randcv = RandomizedSearchCV(rfc, param_distributions, n_iter=100, random_state=0, cv=5)
    randcv.fit(X_train, y_train)
    
    # final training
    train_and_save_final_model(X, y, randcv.best_params_, save_model_file_path)
    
    # Test Data Accuracy Score
    y_test_pred = randcv.best_estimator_.predict(X_test)
    best_params = randcv.best_params_
    test_data_accuracy_score = accuracy_score(y_test, y_test_pred)
    return best_params, test_data_accuracy_score
