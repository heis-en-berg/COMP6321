from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, ExpSineSquared
from sklearn.model_selection import RandomizedSearchCV
import pickle

def train_and_save_final_model(X, y, params, save_model_file_path):
    gpr=GaussianProcessRegressor(random_state=0)
    gpr.set_params(**params)
    gpr.fit(X, y)
    
    #save model
    model_file_path = save_model_file_path + 'gpr.sav'
    pickle.dump(gpr, open(model_file_path, 'wb'))

def fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path):
    # cross validation for hyperparameter tuning
    param_distributions = {
            'kernel': [RBF(), ConstantKernel()]
            }
    gpr=GaussianProcessRegressor(random_state=0)
    randcv = RandomizedSearchCV(gpr, param_distributions, random_state=0, cv=5)
    randcv.fit(X_train, y_train)
    
    # final training
    train_and_save_final_model(X, y, randcv.best_params_, save_model_file_path)

