import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from timeout_decorator import timeout
from timeout_decorator.timeout_decorator import TimeoutError

def load_and_preprocess_data(filepath, filename):
    # read data (dataset at "http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime")
    data = np.loadtxt(filepath + filename, delimiter=',',dtype='str')
    X_old=np.concatenate((data[:,5:101],data[:,118:121]),axis=1)
    X_old=np.concatenate((X_old,data[:,125:126]),axis=1)
    y=data[:,127:].reshape(-1,)
    
    letters = {'?':'0'}
    X=np.copy(X_old)
    for old, new in letters.items():
        X[X_old==old] = new
    
    X=X.astype(np.float32)
    y=y.astype(np.float32)
    
    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    # scaling data
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)
    
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1,1)).reshape(-1)
    
    np.save(filepath + 'X_train.npy', X_train, allow_pickle=True)
    np.save(filepath + 'X_test.npy', X_test, allow_pickle=True)
    np.save(filepath + 'y_train.npy', y_train, allow_pickle=True)
    np.save(filepath + 'y_test.npy', y_test, allow_pickle=True)
    
    return X, y, X_train, X_test, y_train, y_test


def fit_and_tune_models(model, X, y, X_train, X_test, y_train, y_test, save_model_file_path, test_data):
    model.fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path, test_data)


def call_with_timeout(seconds, f, *args, **kwargs):

    # Define a function that forwards the arguments but times out
    @timeout(seconds)
    def f_timeout(*args, **kwargs):
        return f(*args, **kwargs)

    # Call the timeout wrapper function
    try:
        return f_timeout(*args, **kwargs)
    except TimeoutError as e:
        print(e)
        print("Error: function timed")

def evaluate_models(filepath, filename, list_of_models, save_model_file_path, TIMEOUT, test_data):
    X, y, X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath, filename)
    
    for model in list_of_models:
        print(model)
        call_with_timeout(TIMEOUT, fit_and_tune_models, model, X, y, X_train, X_test, y_train, y_test, save_model_file_path, test_data)
