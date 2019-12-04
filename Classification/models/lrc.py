from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import scipy.stats
import pickle

def train_and_save_final_model(X, y, params, save_model_file_path):
    lrc=LogisticRegression(random_state=0, n_jobs=-1)
    lrc.set_params(**params)
    lrc.fit(X, y)
    
    #save model
    model_file_path = save_model_file_path + 'lrc.sav'
    pickle.dump(lrc, open(model_file_path, 'wb'))

def fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path):
    # cross validation for hyperparameter tuning
    param_distributions = {
            'C': scipy.stats.reciprocal(0.1, 2.5),
            'max_iter': scipy.stats.randint(200,300)
            }
    lrc=LogisticRegression(random_state=0, n_jobs=-1)
    randcv = RandomizedSearchCV(lrc, param_distributions, n_iter=50, random_state=0, cv=5)
    randcv.fit(X_train, y_train)
    
    # final training
    train_and_save_final_model(X, y, randcv.best_params_, save_model_file_path)
    
    # Test Data Accuracy Score
    y_test_pred = randcv.best_estimator_.predict(X_test)
    best_params = randcv.best_params_
    test_data_accuracy_score = accuracy_score(y_test, y_test_pred)
    return best_params, test_data_accuracy_score
