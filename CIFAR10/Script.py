from CIFAR10.preprocess_and_eval_model import evaluate_models as CIFAR10
import CIFAR10.models.dtc as dtc
import CIFAR10.models.cnn as cnn
import CIFAR10.models.am as am
import os

list_of_models = [dtc, cnn, am]
current_directory = os.getcwd()
TIMEOUT = 600.0 # seconds

try:
    # Decision Tree on CIFAR-10
    CIFAR10("./data/", 
                    list_of_models, 
                    current_directory + '/saved_models/', 
                    TIMEOUT)
except:
    print("Something went wrong while evaluating models for dataset: CIFAR-10")
    