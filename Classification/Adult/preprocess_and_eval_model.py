import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from timeout_decorator import timeout
from timeout_decorator.timeout_decorator import TimeoutError

def load_and_preprocess_data(filepath, filename):
    # Reading and Encoding categorical features
    categorcal_features = [1,3,5,6,7,8,9,13]
    non_categorical_features = [0,2,4,10,11,12]
    
    class_column = 14
    
    X_categorical_features = np.loadtxt(filepath + filename, delimiter=',',dtype='str', usecols=categorcal_features)
    X_non_categorical_features = np.loadtxt(filepath + filename, delimiter=',', dtype=np.int32, usecols=non_categorical_features)
    y = np.loadtxt(filepath + filename, delimiter=',', dtype=np.str, usecols=class_column)
    
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(X_categorical_features)
    X_categorical_features = one_hot_encoder.transform(X_categorical_features).toarray()
    
    X = np.concatenate((X_non_categorical_features, X_categorical_features), axis=1)
        
    unique_classes = np.unique(y)
    le = LabelEncoder()
    le.fit(unique_classes)
    y = le.transform(y)
    
    # split data into training and test set
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    
    # preprocessing - scaling data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X = scaler.transform(X)
    
    np.save(filepath + 'X_train.npy', X_train, allow_pickle=True)
    np.save(filepath + 'X_test.npy', X_test, allow_pickle=True)
    np.save(filepath + 'y_train.npy', y_train, allow_pickle=True)
    np.save(filepath + 'y_test.npy', y_test, allow_pickle=True)
    joblib.dump(one_hot_encoder, filepath + 'one_hot_encoder.joblib')
    joblib.dump(scaler, filepath + 'scaler.joblib')
    
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


def load_test_data(filepath, filename):
    if filepath == None:
        X_test = np.load(filepath + "X_test.npy", allow_pickle=True)
        y_test = np.load(filepath + "y_test.npy", allow_pickle=True)
        return X_test, y_test
    else:
        categorcal_features = [1,3,5,6,7,8,9,13]
        non_categorical_features = [0,2,4,10,11,12]
        class_column = 14
        
        X_categorical_features = np.loadtxt(filepath + filename, delimiter=',',dtype='str', usecols=categorcal_features, skiprows=1)
        X_non_categorical_features = np.loadtxt(filepath + filename, delimiter=',', dtype=np.int32, usecols=non_categorical_features, skiprows=1)
        y_test = np.loadtxt(filepath + filename, delimiter=',', dtype=np.str, usecols=class_column, skiprows=1)
        
        one_hot_encoder = joblib.load(filepath + "one_hot_encoder.joblib")
        X_categorical_features = one_hot_encoder.transform(X_categorical_features).toarray()
        
        X_test = np.concatenate((X_non_categorical_features, X_categorical_features), axis=1)
            
        unique_classes = np.unique(y_test)
        le = LabelEncoder()
        le.fit(unique_classes)
        y_test = le.transform(y_test)
        
        # scaling data
        scaler = joblib.load(filepath + "scaler.joblib")
        X_test = scaler.transform(X_test)
        
        return X_test, y_test