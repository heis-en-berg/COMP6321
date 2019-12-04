from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle

def train_and_save_final_model(X, y, params, save_model_file_path):
    mlp=MLPClassifier(random_state=0)
    mlp.set_params(**params)
    mlp.fit(X, y)
    
    #save model
    model_file_path = save_model_file_path + 'mlp.sav'
    pickle.dump(mlp, open(model_file_path, 'wb'))

def fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path):
    # cross validation for hyperparameter tuning
    param_distributions = {
            'hidden_layer_sizes': [(100,50,), (100,50,20,)],
            'learning_rate': ["constant", "invscaling", "adaptive"],
            'max_iter': [200, 250]
            }
    mlp=MLPClassifier(random_state=0)
    randcv = RandomizedSearchCV(mlp, param_distributions, random_state=0, cv=5)
    randcv.fit(X_train, y_train)
    
    # final training
    train_and_save_final_model(X, y, randcv.best_params_, save_model_file_path)
