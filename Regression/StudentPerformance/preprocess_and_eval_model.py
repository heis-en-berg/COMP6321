import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from timeout_decorator import timeout
from timeout_decorator.timeout_decorator import TimeoutError

def load_and_preprocess_data(filename):
    # read and encode data (dataset at "http://archive.ics.uci.edu/ml/datasets/Student+Performance")
    categorcal_features = [0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22]
    non_categorical_features = [2,6,7,12,13,14,23,24,25,26,27,28,29]
    class_column = 32
    
    X_categorical_features = np.loadtxt(filename, delimiter=';',dtype='str', usecols=categorcal_features, skiprows=1)
    X_non_categorical_features = np.loadtxt(filename, delimiter=';', 
                                            dtype=np.int32, 
                                            usecols=non_categorical_features, 
                                            skiprows=1)
    def decode_str(s):
        return int(s.decode("utf-8").replace('"', ''))
                   
    g1 = np.genfromtxt(filename, delimiter=';', 
                                    dtype='str', 
                                    skip_header=True,
                                    usecols=[30],
                                    converters = {30: decode_str}).reshape(-1,1)
    
    g2 = np.genfromtxt(filename, delimiter=';', 
                                    dtype='str', 
                                    skip_header=True,
                                    usecols=[31],
                                    converters = {31: decode_str}).reshape(-1,1)
    y = np.loadtxt(filename, delimiter=';', dtype=np.int64, usecols=class_column, skiprows=1)
    
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(X_categorical_features)
    X_categorical_features = one_hot_encoder.transform(X_categorical_features).toarray()
    
    X = np.concatenate((X_non_categorical_features, X_categorical_features), axis=1)
    X = np.concatenate((X, g1), axis=1)
    X = np.concatenate((X, g2), axis=1)
    
    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    # scaling data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).reshape(-1)
    y_test = y_scaler.fit_transform(y_test.reshape(-1,1)).reshape(-1)
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
