from Classification.BreastCancerWinsconsin.preprocess_and_eval_model import evaluate_models as breast_cancer_winsconsin_eval_models
from Classification.DefaultOfCreditCardClients.preprocess_and_eval_model import evaluate_models as default_credit_card_eval_models
from Classification.DiabeticRetinopathy.preprocess_and_eval_model import evaluate_models as diabetic_retinopathy_eval_models
from Classification.StatlogAustralianCreditApproval.preprocess_and_eval_model import evaluate_models as statlog_australian__eval_models
from Classification.StatlogGermanCreditData.preprocess_and_eval_model import evaluate_models as statog_german_eval_models
from Classification.Adult.preprocess_and_eval_model import evaluate_models as adult_eval_models
from Classification.SteelPlatesFaults.preprocess_and_eval_model import evaluate_models as steel_plates_eval_models
from Classification.Yeast.preprocess_and_eval_model import evaluate_models as yeast_eval_models
from Classification.ThoracicSurgeryData.preprocess_and_eval_model import evaluate_models as thoraric_eval_models
from Classification.SeismicBumps.preprocess_and_eval_model import evaluate_models as sesamic_bumps_eval_models

from sklearn.exceptions import ConvergenceWarning
import warnings
import os

import Classification.models.dtc as dtc
import Classification.models.adbc as adbc
import Classification.models.gnbc as gnbc
import Classification.models.knn as knn
import Classification.models.lrc as lrc
import Classification.models.mlp as mlp
import Classification.models.rfc as rfc
import Classification.models.svc as svc

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def classification_train_models(TIMEOUT):
    current_directory = os.getcwd()
    list_of_models = [dtc, adbc, gnbc, knn, lrc, mlp, rfc, svc]
    
    try:
        # Breast Cancer Winsconsin
        print("Breast Cancer Winsconsin dataset")
        breast_cancer_winsconsin_eval_models("./BreastCancerWinsconsin/data/", "wdbc.data", 
                        list_of_models, 
                        current_directory + '/BreastCancerWinsconsin/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: Breast Cancer Winsconsin")
    
    try:
        # Default Of Credit Card Clients
        print("Default Of Credit Card Clients dataset")
        default_credit_card_eval_models("./DefaultOfCreditCardClients/data/","data.xls", 
                        list_of_models, 
                        current_directory + '/DefaultOfCreditCardClients/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: Default Of Credit Card Clients")
    
    try:
        # Diabetic Retinopathy
        print("Diabetic Retinopathy dataset")
        diabetic_retinopathy_eval_models("./DiabeticRetinopathy/data/", "messidor_features.arff", 
                        list_of_models, 
                        current_directory + '/DiabeticRetinopathy/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: Diabetic Retinopathy")
    
    try:
        # StatlogAustralianCreditApproval
        print("StatlogAustralianCreditApproval dataset")
        statlog_australian__eval_models("./StatlogAustralianCreditApproval/data/", "australian.dat", 
                        list_of_models, 
                        current_directory + '/StatlogAustralianCreditApproval/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: StatlogAustralianCreditApproval")
    
    try:
        # StatlogGermanCreditData
        print("StatlogGermanCreditData dataset")
        statog_german_eval_models("./StatlogGermanCreditData/data/", "german.data-numeric", 
                        list_of_models, 
                        current_directory + '/StatlogGermanCreditData/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: StatlogGermanCreditData")
    
    try:
        # Adult
        print("Adult dataset")
        adult_eval_models("./Adult/data/", "adult.data", 
                        list_of_models, 
                        current_directory + '/Adult/saved_models/', 
                        TIMEOUT, "./Adult/data/adult.test")
    except:
        print("Something went wrong while evaluating models for dataset: Adult")
    
    try:
        # SteelPlatesFaults
        print("SteelPlatesFaults dataset")
        steel_plates_eval_models("./SteelPlatesFaults/data/", "Faults.NNA", 
                        list_of_models, 
                        current_directory + '/SteelPlatesFaults/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: SteelPlatesFaults")
    
    try:
        # Yeast
        print("Yeast dataset")
        yeast_eval_models("./Yeast/data/", "yeast.data", 
                        list_of_models, 
                        current_directory + '/Yeast/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: Yeast")
    
    try:
        # ThoracicSurgeryData
        print("ThoracicSurgeryData dataset")
        thoraric_eval_models("./ThoracicSurgeryData/data/", "ThoraricSurgery.arff", 
                        list_of_models, 
                        current_directory + '/ThoracicSurgeryData/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: ThoracicSurgeryData")
    
    try:
        # SeismicBumps
        print("SeismicBumps dataset")
        sesamic_bumps_eval_models("./SeismicBumps/data/", "seismic-bumps.arff", 
                        list_of_models, 
                        current_directory + '/SeismicBumps/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: SeismicBumps")