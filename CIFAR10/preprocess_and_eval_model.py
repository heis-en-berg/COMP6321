import numpy as np
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from timeout_decorator import timeout
from timeout_decorator.timeout_decorator import TimeoutError

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_and_preprocess_data(path):
    data=unpickle(path+"data_batch_1")
    X=np.array(data[b'data']).astype(np.uint8)
    y=np.array(data[b'labels']).astype(np.int32)
    for i in range(4):
        data=unpickle(path+"data_batch_"+str(i+2))
        X=np.vstack((X,np.array(data[b'data'])))
        y=np.vstack((y,np.array(data[b'labels'])))
    y=y.reshape(-1,)
    X_train=X
    y_train=y
    
    data=unpickle(path+"test_batch")
    X_test=np.array(data[b'data']).astype(np.uint8)
    y_test=np.array(data[b'labels']).astype(np.int32)
    
    scaler=sklearn.preprocessing.StandardScaler().fit(X)
    X_train=scaler.transform(X)
    X_test=scaler.transform(X_test)
    return X, y, X_train, X_test, y_train, y_test

def fit_and_tune_models(model, X, y, X_train, X_test, y_train, y_test, save_model_file_path):
    model.fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path)


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

def evaluate_models(filename, list_of_models, save_model_file_path, TIMEOUT):
    X, y, X_train, X_test, y_train, y_test = load_and_preprocess_data(filename)
    
    for model in list_of_models:
        print(model)
        call_with_timeout(TIMEOUT, fit_and_tune_models, model, X, y, X_train, X_test, y_train, y_test, save_model_file_path)

