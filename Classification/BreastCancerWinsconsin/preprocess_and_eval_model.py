import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from timeout_decorator import timeout
from timeout_decorator.timeout_decorator import TimeoutError

def load_and_preprocess_data(filename):
    # read data (dataset at "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)")
    feature_columns_index = np.arange(2,32)
    label_column_index = 1
    X = np.loadtxt(filename, delimiter=',', usecols = feature_columns_index)
    y = np.loadtxt(filename, delimiter=',', usecols = label_column_index, dtype=np.str)
    
    # preprocessing
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
