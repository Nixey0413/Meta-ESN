import torch
import time
import os
import numpy as np
import scipy.io as sio
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load target task dataset
# =============================================================================
X_in = torch.load('Reptile/X_in_entire.pth', weights_only=False)
Y_in = torch.load('Reptile/Y_in_entire.pth', weights_only=False)
X_te = torch.load('Reptile/X_te_entire.pth', weights_only=False)
Y_te = torch.load('Reptile/Y_te_entire.pth', weights_only=False)

X_full = torch.load('Reptile/X_full_0.pth', weights_only=False)
Y_full = torch.load('Reptile/Y_full_0.pth', weights_only=False)
print('Loading completed')
# =============================================================================

# Batch size
Num_bat = 10
# Learning rate
Learn_r = 0.0001
# Epoch
epoch = 100
# Target task
target = 0
# Few-shot parameter
alpha_list = [1/4, 0.75/4, 0.5/4, 0.25/4, 0.05/4]

for alpha in alpha_list:
    # Dataloader
    X_tr, Y_tr = X_in[target,:int(X_in.size(1)*alpha),:,:], Y_in[target,:int(Y_in.size(1)*alpha),:,:]
    My_data = TensorDataset(X_tr.detach(), Y_tr.detach())
    My_loader = DataLoader(My_data, batch_size=Num_bat, shuffle=True)
    X_tar_te = X_te[target, :, :]
    Y_tar_te = Y_te[target, :, :]
    
    # Model setup
    #Test_Model = torch.load('Reptile/Task study/Pre_Model-' + str(target + 1) + '.pkl', weights_only = False).to(device)
    Test_Model = torch.load('Reptile/Pre_Model-entire.pkl', weights_only = False).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(Test_Model.parameters(), lr=Learn_r)
    
    # Gradient function
    def grad(Y):
        Y_grad = torch.stack(((Y[:, 1, 0] - Y[:, 0, 0]),
                              (Y[:, 2, 0] - Y[:, 1, 0])))
        return Y_grad
    
    # Model training
    loss_log = torch.zeros(epoch, dtype=torch.float).to(device)
    test_log = torch.zeros(epoch, dtype=torch.float).to(device)
    Test_Model.train()
    print('Training start')
    for i in range(epoch):
        epoch_start_time = time.time()
        for X, Y in My_loader:
            # Output true value
            Y_hat = Test_Model(X.to(device))
            # Output gradient
            Y_grad_hat = grad(Y_hat)
            Y_grad = grad(Y)
            # Output peak value
            Num_bat = Y_hat.size(0)
            Y_peak_hat = torch.zeros(Num_bat, 1)
            Y_peak = torch.zeros(Num_bat, 1)
            for j in range(Num_bat):
                if (Y_grad[0, j] * Y_grad[1, j]) < 0:
                    Y_peak_hat[j, 0] = Y_hat[j, 1, 0]
                    Y_peak[j, 0] = Y[j, 1, 0]
            Y_peak_hat.require_grad = True
            Y_peak.require_grad = True
            
            # Loss construction
            loss = loss_fn(Y_hat.to(device), Y.to(device))
            loss_p = loss_fn(Y_peak_hat.to(device), Y_peak.to(device))
            loss_g = loss_fn(Y_grad_hat.to(device), Y_grad.to(device))
            loss_all = loss + 5 *loss_p + 0.5*loss_g
    
            # Optimization
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            
        # Training log
        loss_log[i] = loss_fn(Test_Model(X_tr.to(device)), Y_tr.to(device)).detach()
        test_log[i] = loss_fn(Test_Model(X_tar_te.to(device)), Y_tar_te.to(device)).detach()
            
        if i % 10 == 9:
            Predict = Test_Model(X_full.to(device)).to('cpu').detach().squeeze(0)
            plt.figure(figsize=(10, 4))
            plt.plot(Y_full, 'r', label='True', alpha = 0.5)
            plt.plot(Predict, 'b', label='Predict', alpha = 0.5)
            plt.legend(loc='upper right', fontsize=10)
            plt.show()
            
            print('| end of epoch {:3d} | time: {:5.2f}s | Train loss {:5.5f} | Test loss {:5.5f} |'.
                  format(i + 1 , (time.time() - epoch_start_time),
                         1e5 * loss.to('cpu').detach(), 1e5 * test_log[i].to('cpu').detach()))
    
    rep_train = np.array(loss_log.detach().cpu())
    rep_test = np.array(test_log.detach().cpu())
    sio.savemat('Reptile/Reptile_log-' + str(alpha) + '.mat', {'rep_train':rep_train, 'rep_test':rep_test})
    torch.save(Test_Model, 'Reptile/Reptile_Model-' + str(alpha) + '.pkl')



