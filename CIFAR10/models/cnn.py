import scipy
import pickle
import torch
import torch.nn
import torch.optim

def fit_and_tune_model(X, y, X_train, X_test, y_train, y_test, save_model_file_path):
    mean=X_train.mean()
    X_train=X_train-mean  
    
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_train=torch.tensor(X_train)
    y_train=torch.tensor(y_train)
    filter_size = 3
    pool_size = 2
    
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=3,out_channels=32, kernel_size=filter_size,padding=filter_size-2),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(in_channels=32,out_channels=64, kernel_size=filter_size,padding=filter_size-2),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        
        torch.nn.Conv2d(in_channels=64,out_channels=128, kernel_size=filter_size,padding=filter_size-2),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(in_channels=128,out_channels=128, kernel_size=filter_size,padding=filter_size-2),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        torch.nn.Dropout2d(p=0.05),
        
        torch.nn.Conv2d(in_channels=128,out_channels=256, kernel_size=filter_size,padding=filter_size-2),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(in_channels=256,out_channels=256, kernel_size=filter_size,padding=filter_size-2),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        
        torch.nn.Dropout2d(p=0.1),
        torch.nn.Flatten(),
        torch.nn.Linear(256 * 8**2 // pool_size**2, 1024),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(512, 10)
    )
    
    torch.manual_seed(0)
    batch_size = 50
    num_epoch = 22
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
    
    for epoch in range(1, num_epoch+1):
        for i in range(0, len(X_train), batch_size):        
            X = X_train[i:i+batch_size]
            y = y_train[i:i+batch_size]
    
            y_pred = model(X)
            l = loss(y_pred, y)
            
            model.zero_grad()
            l.backward()
            optimizer.step()
            
        print("Epoch %d, batch loss: %.4f" % (epoch, l.item()))
      
    
    #save model
    model_file_path = save_model_file_path + 'cnn.sav'
    pickle.dump(model, open(model_file_path, 'wb'))
    
    #model = torch.load('./saved_models/CNNJ2.pt')
    model.eval()
    correct=0
    y_range=10000
    for i in range(y_range):
        y=model(X_test[i].reshape(1, 3, 32, 32))
        _, index= torch.max(y.data,1)
        if(y_test[i]==index):
            correct+=1
    print("Test accuracy: ",(correct/y_range)*100)

