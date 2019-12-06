import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn

def f(X): 
    model = torch.load('./saved_models/CNNJ2.pt')
    model.eval()
    return model(X) 

def fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path):
    X = np.ones(3072).astype(np.float32)    
    X=torch.zeros(1, 3, 32, 32, requires_grad=True)
    #For category: horse=7, bird=2
    category=7
    for i in range(50):             
        y_pred=f(X)
        y_pred=y_pred[0,category]     
        y_pred.backward()   
        X.data += 0.2*X.grad.data          
        X.grad.data.zero_()          

    new_images = np.transpose(np.reshape(X.detach().numpy(),(3, 32,32)), (1,2,0))
    plt.imshow(new_images)
    plt.show()