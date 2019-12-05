from Regression.BikeSharing.preprocess_and_eval_model import evaluate_models as bike_sharing_eval_models
from Regression.StudentPerformance.preprocess_and_eval_model import evaluate_models as student_performance_card_eval_models
from Regression.ConcreteCompressiveStrength.preprocess_and_eval_model import evaluate_models as concrete_compressive_eval_models
from Regression.SGEMM_GPUKernelPerformance.preprocess_and_eval_model import evaluate_models as gpu_kernel_eval_models
from Regression.MerckMolecularActivity.preprocess_and_eval_model import evaluate_models as merck_moecular_eval_models
from Regression.WineQuality.preprocess_and_eval_model import evaluate_models as wine_quality_eval_models
from Regression.CommunitesAndCrime.preprocess_and_eval_model import evaluate_models as communities_and_crime_eval_models
from Regression.QSARAquaticToxicity.preprocess_and_eval_model import evaluate_models as qsar_eval_models
from Regression.ParkinsonSpeech.preprocess_and_eval_model import evaluate_models as parkinson_speech_eval_models
from Regression.FacebookMetrics.preprocess_and_eval_model import evaluate_models as facebook_metrics_eval_models

from sklearn.exceptions import ConvergenceWarning
import warnings
import os

import Regression.models.dtr as dtr
import Regression.models.adbr as adbr
import Regression.models.gpr as gpr
import Regression.models.lr as lr
import Regression.models.nnr as nnr
import Regression.models.rfr as rfr
import Regression.models.svr as svr

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def regression_train_models(TIMEOUT):
    current_directory = os.getcwd()
    
    list_of_models = [dtr, adbr, gpr, lr, nnr, rfr, svr]
    
    try:
        # BikeSharing
        print("BikeSharing dataset")
        bike_sharing_eval_models("./BikeSharing/data/" , "hour.csv", 
                        list_of_models, 
                        current_directory + '/BikeSharing/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: BikeSharing")
        
    try:
        # StudentPerformance
        print("StudentPerformance dataset")
        student_performance_card_eval_models("./StudentPerformance/data/" , "student-por.csv", 
                        list_of_models, 
                        current_directory + '/StudentPerformance/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: StudentPerformance")
    
    try:
        # ConcreteCompressiveStrength
        print("ConcreteCompressiveStrength dataset")
        concrete_compressive_eval_models("./ConcreteCompressiveStrength/data/" , "Concrete_Data.xls", 
                        list_of_models, 
                        current_directory + '/ConcreteCompressiveStrength/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: ConcreteCompressiveStrength")
    
    try:
        # WineQuality
        print("WineQuality dataset")
        wine_quality_eval_models("./WineQuality/data/" , "winequality-red.csv", 
                        list_of_models, 
                        current_directory + '/WineQuality/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: WineQuality")
    
    try:
        # CommunitesAndCrime
        print("CommunitesAndCrime dataset")
        communities_and_crime_eval_models("./CommunitesAndCrime/data/" , "communities.data", 
                        list_of_models, 
                        current_directory + '/CommunitesAndCrime/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: CommunitesAndCrime")
    
    try:
        # QSARAquaticToxicity
        print("QSARAquaticToxicity dataset")
        qsar_eval_models("./QSARAquaticToxicity/data/" , "qsar_aquatic_toxicity.csv", 
                        list_of_models, 
                        current_directory + '/QSARAquaticToxicity/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: QSARAquaticToxicity")
    
    try:
        # ParkinsonSpeech
        print("ParkinsonSpeech dataset")
        parkinson_speech_eval_models("./ParkinsonSpeech/data/" , "train_data.txt", 
                        list_of_models, 
                        current_directory + '/ParkinsonSpeech/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: ParkinsonSpeech")
    
    try:
        # FacebookMetrics
        print("FacebookMetrics dataset")
        facebook_metrics_eval_models("./FacebookMetrics/data/" , "dataset_Facebook.csv", 
                        list_of_models, 
                        current_directory + '/FacebookMetrics/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: FacebookMetrics")
    
    try:
        # SGEMM_GPUKernelPerformance
        print("SGEMM_GPUKernelPerformance dataset")
        gpu_kernel_eval_models("./SGEMM_GPUKernelPerformance/data/" , "sgemm_product.csv", 
                        list_of_models, 
                        current_directory + '/SGEMM_GPUKernelPerformance/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: SGEMM_GPUKernelPerformance")
    
    
    try:
        # MerckMolecularActivity ACT2_competition_training
        print("MerckMolecularActivity ACT2_competition_training dataset")
        merck_moecular_eval_models("./MerckMolecularActivity/data/" , "ACT2_competition_training.csv", 
                        list_of_models, 
                        current_directory + '/MerckMolecularActivity/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: MerckMolecularActivity ACT2_competition_training")
    
    try:
        # MerckMolecularActivity ACT4_competition_training
        print("MerckMolecularActivity ACT4_competition_training dataset")
        merck_moecular_eval_models("./MerckMolecularActivity/data/" , "ACT4_competition_training.csv", 
                        list_of_models, 
                        current_directory + '/MerckMolecularActivity/saved_models/', 
                        TIMEOUT, None)
    except:
        print("Something went wrong while evaluating models for dataset: MerckMolecularActivity ACT4_competition_training")
    
    # remove jibble
    # refactor master_scrpt's evaluation code
    # point 4