from BreastCancerWinsconsin.preprocess_and_eval_model import evaluate_models as breast_cancer_winsconsin_eval_models
from DefaultOfCreditCardClients.preprocess_and_eval_model import evaluate_models as default_credit_card_eval_models
from DiabeticRetinopathy.preprocess_and_eval_model import evaluate_models as diabetic_retinopathy_eval_models
from StatlogAustralianCreditApproval.preprocess_and_eval_model import evaluate_models as statlog_australian__eval_models
from StatlogGermanCreditData.preprocess_and_eval_model import evaluate_models as statog_german_eval_models
from Adult.preprocess_and_eval_model import evaluate_models as adult_eval_models
from SteelPlatesFaults.preprocess_and_eval_model import evaluate_models as steel_plates_eval_models
from Yeast.preprocess_and_eval_model import evaluate_models as yeast_eval_models
from ThoracicSurgeryData.preprocess_and_eval_model import evaluate_models as thoraric_eval_models
from SeismicBumps.preprocess_and_eval_model import evaluate_models as sesamic_bumps_eval_models

from sklearn.exceptions import ConvergenceWarning
import warnings
import os

import models.dtc as dtc
import models.adbc as adbc
import models.gnbc as gnbc
import models.knn as knn
import models.lrc as lrc
import models.mlp as mlp
import models.rfc as rfc
import models.svc as svc

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

current_directory = os.getcwd()

TIMEOUT = 3600.0 # seconds
list_of_models = [dtc, adbc, gnbc, knn, lrc, mlp, rfc, svc]

try:
    # Breast Cancer Winsconsin
    breast_cancer_winsconsin_eval_models("./BreastCancerWinsconsin/data/wdbc.data", 
                    list_of_models, 
                    current_directory + '/BreastCancerWinsconsin/saved_models/', 
                    TIMEOUT)
except:
    print("Something went wrong while evaluating models for dataset: Breast Cancer Winsconsin")

try:
    # Default Of Credit Card Clients
    default_credit_card_eval_models("./DefaultOfCreditCardClients/data/data.xls", 
                    list_of_models, 
                    current_directory + '/DefaultOfCreditCardClients/saved_models/', 
                    TIMEOUT)
except:
    print("Something went wrong while evaluating models for dataset: Default Of Credit Card Clients")

try:
    # Diabetic Retinopathy
    diabetic_retinopathy_eval_models("./DiabeticRetinopathy/data/messidor_features.arff", 
                    list_of_models, 
                    current_directory + '/DiabeticRetinopathy/saved_models/', 
                    TIMEOUT)
except:
    print("Something went wrong while evaluating models for dataset: Diabetic Retinopathy")

try:
    # StatlogAustralianCreditApproval
    statlog_australian__eval_models("./StatlogAustralianCreditApproval/data/australian.dat", 
                    list_of_models, 
                    current_directory + '/StatlogAustralianCreditApproval/saved_models/', 
                    TIMEOUT)
except:
    print("Something went wrong while evaluating models for dataset: StatlogAustralianCreditApproval")

try:
    # StatlogGermanCreditData
    statog_german_eval_models("./StatlogGermanCreditData/data/german.data-numeric", 
                    list_of_models, 
                    current_directory + '/StatlogGermanCreditData/saved_models/', 
                    TIMEOUT)
except:
    print("Something went wrong while evaluating models for dataset: StatlogGermanCreditData")

try:
    # Adult
    adult_eval_models("./Adult/data/adult.data", 
                    list_of_models, 
                    current_directory + '/Adult/saved_models/', 
                    TIMEOUT)
except:
    print("Something went wrong while evaluating models for dataset: Adult")

try:
    # SteelPlatesFaults
    steel_plates_eval_models("./SteelPlatesFaults/data/Faults.NNA", 
                    list_of_models, 
                    current_directory + '/SteelPlatesFaults/saved_models/', 
                    TIMEOUT)
except:
    print("Something went wrong while evaluating models for dataset: SteelPlatesFaults")

try:
    # Yeast
    yeast_eval_models("./Yeast/data/yeast.data", 
                    list_of_models, 
                    current_directory + '/Yeast/saved_models/', 
                    TIMEOUT)
except:
    print("Something went wrong while evaluating models for dataset: Yeast")

try:
    # ThoracicSurgeryData
    thoraric_eval_models("./ThoracicSurgeryData/data/ThoraricSurgery.arff", 
                    list_of_models, 
                    current_directory + '/ThoracicSurgeryData/saved_models/', 
                    TIMEOUT)
except:
    print("Something went wrong while evaluating models for dataset: ThoracicSurgeryData")

try:
    # SeismicBumps
    sesamic_bumps_eval_models("./SeismicBumps/data/seismic-bumps.arff", 
                    list_of_models, 
                    current_directory + '/SeismicBumps/saved_models/', 
                    TIMEOUT)
except:
    print("Something went wrong while evaluating models for dataset: SeismicBumps")