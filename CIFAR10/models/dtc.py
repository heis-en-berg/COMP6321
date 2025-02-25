from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import scipy
import pickle

def train_and_save_final_model(X, y, X_train, X_test, y_train, y_test, params, save_model_file_path):
    dtc=DecisionTreeClassifier(random_state=0)
    dtc.set_params(**params)
    dtc.fit(X, y)
    
    y_test_pred = dtc.predict(X_test)
    print("Accuracy score for Decision Tree (CIFAR-10): ", accuracy_score(y_test, y_test_pred))
    
    #save model
    model_file_path = save_model_file_path + 'dtc.sav'
    pickle.dump(dtc, open(model_file_path, 'wb'))
    

def fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path):
    # cross validation for hyperparameter tuning
    param_distributions = {
            #'max_depth': scipy.stats.randint(1,30),
            #'criterion': ["gini", "entropy"]
            }
    dtc=DecisionTreeClassifier(random_state=0)
    gridcv=GridSearchCV(dtc, param_distributions, verbose=1, cv=5)
    gridcv.fit(X_train, y_train)
    #randcv = RandomizedSearchCV(dtc, param_distributions, random_state=0, cv=5)
    #randcv.fit(X_train, y_train)
    
    # final training
    train_and_save_final_model(X, y, X_train, X_test, y_train, y_test, gridcv.best_params_, save_model_file_path)
