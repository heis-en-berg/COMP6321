import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from timeout_decorator import timeout
from timeout_decorator.timeout_decorator import TimeoutError

# indices
indices_path = "./SGEMM_GPUKernelPerformance/indices.npy"
# saved using
'''
indices = np.random.randint(0,241601,12080)
np.save('indices.npy', indices, allow_pickle=True)
'''

def load_and_preprocess_data(filepath, filename):
    # read data (dataset at "http://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance")
    data = np.loadtxt(filepath + filename, delimiter=',', skiprows=1)
    indices = np.load(indices_path, allow_pickle=True)
    
    X = data[indices,:14]
    y = data[indices,14:18]
    
    X_categorical_features = X[:,10:14]
    encoder = OneHotEncoder()
    encoder.fit(X_categorical_features)
    X_categorical_features = encoder.transform(X_categorical_features).toarray()
    
    X = np.delete(X,[10,11,12,13],axis=1)
    X = np.concatenate((X,X_categorical_features), axis=1)
    
    # averaging values row wise in y
    y = y.mean(axis=1)
    
    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    # scaling data
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.fit_transform(X_test)
    
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