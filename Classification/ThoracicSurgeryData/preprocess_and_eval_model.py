import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from timeout_decorator import timeout
from timeout_decorator.timeout_decorator import TimeoutError

def load_and_preprocess_data(filepath, filename):
    # read data (dataset at "https://archive.ics.uci.edu/ml/datasets/Thoracic+Surgery+Data")
    categorcal_features = np.concatenate(([0], np.arange(3,15)))
    non_categorical_features = [1,2,15]
    class_column = 16
    X_categorical_features = np.loadtxt(filepath + filename, delimiter=',', skiprows=21, dtype=np.str, usecols=categorcal_features)
    X_non_categorical_features = np.loadtxt(filepath + filename, delimiter=',', skiprows=21, dtype=np.float64, usecols=non_categorical_features)
    y = np.loadtxt(filepath + filename, delimiter=',', skiprows=21, dtype=np.str, usecols=class_column)
    
    # Encoding data
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(X_categorical_features)
    X_categorical_features = one_hot_encoder.transform(X_categorical_features).toarray()
    
    X = np.concatenate((X_non_categorical_features, X_categorical_features), axis=1)
    
    unique_labels = np.unique(y)
    le = preprocessing.LabelEncoder()
    le.fit(unique_labels)
    y = le.transform(y)
    
    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    # preprocessing - scaling data and features removal
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
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