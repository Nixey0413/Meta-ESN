import torch
import model
import time
import os
import numpy as np
import scipy.io as sio
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load dataset
Case = 'Baseline'
X_tr = torch.load(Case + '/X_tr.pth', weights_only=False)
X_te = torch.load(Case + '/X_te.pth', weights_only=False)
Y_tr = torch.load(Case + '/Y_tr.pth', weights_only=False)
Y_te = torch.load(Case + '/Y_te.pth', weights_only=False)

X_full = torch.load(Case + '/X_full.pth', weights_only=False).unsqueeze(0)
Y_full = torch.load(Case + '/Y_full.pth', weights_only=False)

# Batch size
Num_bat = 10
# Learning rate
Learn_r = 0.0001
# Epoch
epoch = 100
# Few-shot parameter
alpha_list = [1]

for alpha in alpha_list:
    # Dividing batch
    X_in = torch.zeros(int(X_tr.size(0)*alpha - 2), 3, X_tr.size(1))
    Y_in = torch.zeros(int(X_tr.size(0)*alpha - 2), 3, 1)
    for i in range(X_in.size(0)):
        X_in[i, :, :] = X_tr[i: i + 3, :]
        Y_in[i, :, 0] = Y_tr[i: i + 3]
    
    # Dataloader
    My_dataset = TensorDataset(X_in.detach(), Y_in.detach())
    My_loader = DataLoader(My_dataset, batch_size=Num_bat, shuffle=True)
    
    # Model setup
    Dimension = X_tr.size(1)
    Model = model.LSTM().to(device)
    Model.train()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(Model.parameters(), lr=Learn_r)
    
    # Model training
    loss_log = torch.zeros(epoch, dtype=torch.float)
    test_log = torch.zeros(epoch, dtype=torch.float)
    for i in range(epoch):
        epoch_start_time = time.time()
        Model.train()
        
        for X, Y in My_loader:
            # Output true value
            Y_hat = Model(X.to(device))
            # Loss construction
            loss = loss_fn(Y_hat.to(device), Y.to(device))   
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Training log
        with torch.no_grad():
            loss_log[i] = loss_fn(Model(X_tr.to(device)).squeeze(1), Y_tr.to(device)).detach()
            test_log[i] = loss_fn(Model(X_te.to(device)).squeeze(1), Y_te.to(device)).detach()
        
        if i % 10 == 9:
            Model.eval()
            with torch.no_grad():
                Predict = Model(X_full.to(device)).to('cpu').detach().squeeze(0)
                plt.figure(figsize=(10, 4))
                plt.plot(Y_full, 'r', label='True', alpha = 0.5)
                plt.plot(Predict, 'b', label='Predict', alpha = 0.5)
                plt.legend(loc='upper right', fontsize = 10)
                plt.show()
                
                test_log[i] = loss_fn(Model(X_te.to(device)).squeeze(1), Y_te.to(device)).detach()
                print('| end of epoch {:3d} | time: {:5.2f}s | Train loss {:5.5f} | Test loss {:5.5f} |'.
                      format(i + 1, (time.time() - epoch_start_time),
                             1e5 * loss.to('cpu').detach(), 1e5 * test_log[i].to('cpu').detach()))
            Model.train()
    
    Method = 'LSTM'    
    train = np.array(loss_log.detach().cpu())
    test = np.array(test_log.detach().cpu())
    sio.savemat('Baseline/'+ Method +'_log.mat', {Method +'_train':train, Method +'_test':test})
    torch.save(Model, 'Baseline/'+ Method +'_Model.pkl')
    
    pre = np.array(Predict.detach().cpu())
    tru = np.array(Y_full.detach().cpu())
    sio.savemat(Method +'.mat', {'pre':pre, 'tru':tru})  

