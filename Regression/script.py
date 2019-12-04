from BikeSharing.preprocess_and_eval_model import evaluate_models as bike_sharing_eval_models
from StudentPerformance.preprocess_and_eval_model import evaluate_models as student_performance_card_eval_models
from ConcreteCompressiveStrength.preprocess_and_eval_model import evaluate_models as concrete_compressive_eval_models
from SGEMM_GPUKernelPerformance.preprocess_and_eval_model import evaluate_models as gpu_kernel_eval_models
from MerckMolecularActivity.preprocess_and_eval_model import evaluate_models as merck_moecular_eval_models
from WineQuality.preprocess_and_eval_model import evaluate_models as wine_quality_eval_models
from CommunitesAndCrime.preprocess_and_eval_model import evaluate_models as communities_and_crime_eval_models
from QSARAquaticToxicity.preprocess_and_eval_model import evaluate_models as qsar_eval_models
from ParkinsonSpeech.preprocess_and_eval_model import evaluate_models as parkinson_speech_eval_models
from FacebookMetrics.preprocess_and_eval_model import evaluate_models as facebook_metrics_eval_models

from sklearn.exceptions import ConvergenceWarning
import warnings
import os

import models.dtr as dtr
import models.adbr as adbr
import models.gpr as gpr
import models.lr as lr
import models.nnr as nnr
import models.rfr as rfr
import models.svr as svr

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

current_directory = os.getcwd()

TIMEOUT = 60.0 # seconds
list_of_models = [dtr, adbr, gpr, lr, nnr, rfr, svr]

# Breast Cancer Winsconsin
bike_sharing_eval_models("./BikeSharing/data/hour.csv", 
                list_of_models, 
                current_directory + '/BikeSharing/saved_models/', 
                TIMEOUT)

# Default Of Credit Card Clients
student_performance_card_eval_models("./StudentPerformance/data/student-por.csv", 
                list_of_models, 
                current_directory + '/StudentPerformance/saved_models/', 
                TIMEOUT)

# ConcreteCompressiveStrength
concrete_compressive_eval_models("./ConcreteCompressiveStrength/data/Concrete_Data.xls", 
                list_of_models, 
                current_directory + '/ConcreteCompressiveStrength/saved_models/', 
                TIMEOUT)

# WineQuality
wine_quality_eval_models("./WineQuality/data/winequality-red.csv", 
                list_of_models, 
                current_directory + '/WineQuality/saved_models/', 
                TIMEOUT)

# CommunitesAndCrime
communities_and_crime_eval_models("./CommunitesAndCrime/data/communities.data", 
                list_of_models, 
                current_directory + '/CommunitesAndCrime/saved_models/', 
                TIMEOUT)

# QSARAquaticToxicity
qsar_eval_models("./QSARAquaticToxicity/data/qsar_aquatic_toxicity.csv", 
                list_of_models, 
                current_directory + '/QSARAquaticToxicity/saved_models/', 
                TIMEOUT)

# SteelPlatesFaults
parkinson_speech_eval_models("./ParkinsonSpeech/data/train_data.txt", 
                list_of_models, 
                current_directory + '/ParkinsonSpeech/saved_models/', 
                TIMEOUT)

# Yeast
facebook_metrics_eval_models("./FacebookMetrics/data/dataset_Facebook.csv", 
                list_of_models, 
                current_directory + '/FacebookMetrics/saved_models/', 
                TIMEOUT)

# SGEMM_GPUKernelPerformance
gpu_kernel_eval_models("./SGEMM_GPUKernelPerformance/data/sgemm_product.csv", 
                list_of_models, 
                current_directory + '/SGEMM_GPUKernelPerformance/saved_models/', 
                TIMEOUT)


# MerckMolecularActivity
merck_moecular_eval_models("./MerckMolecularActivity/data/ACT2_competition_training.csv", 
                list_of_models, 
                current_directory + '/MerckMolecularActivity/saved_models/', 
                TIMEOUT)

# MerckMolecularActivity
merck_moecular_eval_models("./MerckMolecularActivity/data/ACT4_competition_training.csv", 
                list_of_models, 
                current_directory + '/MerckMolecularActivity/saved_models/', 
                TIMEOUT)


# initializing dataset dictionary with dataset name as keys and filepath as values
datasets = {
        "Diabetic Retinopathy": "./data/wdbc.data",
        "Default of credit card clients": "",
        "Breast Cancer Wisconsin": "",
        "Statlog Australian credit approval)": "",
        "Statlog German credit data)": ""
        }