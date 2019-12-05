import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from Classification.Adult.preprocess_and_eval_model import load_test_data as adult_load_test_data
from Classification.script import classification_train_models
from Regression.script import regression_train_models

TIMEOUT = 3600.0 # seconds
current_directory = os.getcwd()

'''''''''''''''''''Classification'''''''''''''''''''

os.chdir(current_directory + "/Classification")
classification_train_models(TIMEOUT)
os.chdir(current_directory)

list_of__classification_models = ['dtc', 'adbc', 'gnbc', 'knn', 'lrc', 'mlp', 'rfc', 'svc']
result_classification = {}
for model in list_of__classification_models:
    result_classification[model] = []

classification_dataset_paths = [["./Classification/Adult/data/", "./Classification/Adult/saved_models/"],
                        ["./Classification/BreastCancerWinsconsin/data/", "./Classification/BreastCancerWinsconsin/saved_models/"],
                        ["./Classification/DefaultOfCreditCardClients/data/", "./Classification/DefaultOfCreditCardClients/saved_models/"],
                        ["./Classification/DiabeticRetinopathy/data/", "./Classification/DiabeticRetinopathy/saved_models/"],
                        ["./Classification/SeismicBumps/data/", "./Classification/SeismicBumps/saved_models/"],
                        ["./Classification/StatlogAustralianCreditApproval/data/", "./Classification/StatlogAustralianCreditApproval/saved_models/"],
                        ["./Classification/StatlogGermanCreditData/data/", "./Classification/StatlogGermanCreditData/saved_models/"],
                        ["./Classification/SteelPlatesFaults/data/", "./Classification/SteelPlatesFaults/saved_models/"],
                        ["./Classification/ThoracicSurgeryData/data/", "./Classification/ThoracicSurgeryData/saved_models/"],
                        ["./Classification/Yeast/data/", "./Classification/Yeast/saved_models/"]]
      
# load test data and models              
for dataset in classification_dataset_paths:
    data_path = dataset[0]
    model_path = dataset[1]
    try:
        if "Adult" in data_path:
            X_test, y_test = adult_load_test_data(data_path, "adult.test")
        else:
            X_test = np.load(data_path + "X_test.npy", allow_pickle=True)
            y_test = np.load(data_path + "y_test.npy", allow_pickle=True)
        for model in list_of__classification_models:
            try:
                loaded_model = pickle.load(open(model_path + model + '.sav', 'rb'))
                y_pred = loaded_model.predict(X_test)
                acc_score = accuracy_score(y_test, y_pred)
                result_classification[model].append(acc_score)
            except:
                continue
    except:
        continue

print(result_classification)
   
     
'''''''''''''''''''Regression'''''''''''''''''''

os.chdir(current_directory + "/Regression")
regression_train_models(TIMEOUT)
os.chdir(current_directory)

list_of_regression_models = ['adbr', 'dtr', 'gpr', 'lr', 'mlpr', 'rfr', 'svr']
result_regression = {}
for model in list_of_regression_models:
    result_regression[model] = []

regression_dataset_paths = [["./Regression/BikeSharing/data/", "./Regression/BikeSharing/saved_models/"],
                        ["./Regression/CommunitesAndCrime/data/", "./Regression/CommunitesAndCrime/saved_models/"],
                        ["./Regression/ConcreteCompressiveStrength/data/", "./Regression/ConcreteCompressiveStrength/saved_models/"],
                        ["./Regression/FacebookMetrics/data/", "./Regression/FacebookMetrics/saved_models/"],
                        ["./Regression/MerckMolecularActivity/data/", "./Regression/MerckMolecularActivity/saved_models/"],
                        ["./Regression/ParkinsonSpeech/data/", "./Regression/ParkinsonSpeech/saved_models/"],
                        ["./Regression/QSARAquaticToxicity/data/", "./Regression/QSARAquaticToxicity/saved_models/"],
                        ["./Regression/SGEMM_GPUKernelPerformance/data/", "./Regression/SGEMM_GPUKernelPerformance/saved_models/"],
                        ["./Regression/StudentPerformance/data/", "./Regression/StudentPerformance/saved_models/"],
                        ["./Regression/WineQuality/data/", "./Regression/WineQuality/saved_models/"]]
      
# load test data and models              
for dataset in regression_dataset_paths:
    data_path = dataset[0]
    model_path = dataset[1]
    try:
        X_test = np.load(data_path + "X_test.npy", allow_pickle=True)
        y_test = np.load(data_path + "y_test.npy", allow_pickle=True)
        
        for model in list_of_regression_models:
            try:
                loaded_model = pickle.load(open(model_path + model + '.sav', 'rb'))
                y_pred = loaded_model.predict(X_test)
                mean_squared = mean_squared_error(y_test, y_pred)
                result_regression[model].append(mean_squared)
            except:
                continue
    except:
        continue
print(result_regression)