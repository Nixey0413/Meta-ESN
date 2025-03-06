import h5py
import os
import torch
import embedding
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Experiment = h5py.File('Original_data/Data/Exp6&7.mat','r')
Disp = torch.tensor(Experiment ['Disp'], dtype=torch.float32)
Force = torch.tensor(Experiment ['Force'], dtype=torch.float32)
Time = torch.tensor(Experiment ['Time'], dtype=torch.float32)

# Downsample rate
down = 10
# Cycles for training
cycle = 2000
# Reservoir size
Dx = 2000
# Reservoir sparsity
Spar = 1.0
# Reservoir radius
Spec = 0.8
# ESN scale factor
Scale = 0.01
# Reservoir leaky rate
Leaky = 0.6
# Warm-up rate
alpha = 0.005
# Time log
Log = 100
# Time power
Power = 0.9998

# Dataset for tasks
Total_len = int(Disp.size(1)/ down)
Train_len = int(cycle * 100 / down)
Dimension = int(2 * Dx + 5)
X_tr = torch.zeros(8, Train_len, Dimension)
Y_tr = torch.zeros(8, Train_len, 1)
X_te = torch.zeros(8, Total_len - Train_len, Dimension)
Y_te = torch.zeros(8, Total_len - Train_len, 1)
X_full = torch.zeros(Total_len, Dimension)
Y_full = Force[:,::down]

X_full_5 = torch.zeros(Total_len, Dimension)
Y_full_5 = Force[5,::down]

for n in range(8):
    # Downsample dataset
    D = Disp[n,::down] 
    T = Time[n,::down]
    F = Force[n,::down]
    T1, T2 = embedding.decay(D, Log, Power)
    
    # Feature engnieering
    D_peak = torch.zeros(D.size(0))
    D_grad = torch.zeros(D.size(0))
    D_acc = torch.zeros(D.size(0))
    D_grad[0] = D[1] - D[0]
    for i in range(D_peak.size(0) - 2):
        D_grad[i + 1] = D[i + 2] - D[i + 1]
        D_acc[i] = D_grad[i + 1] - D_grad[i]
        if D_grad[i] * D_grad[i + 1] < 0:
            D_peak[i + 1] = D[i + 1]
        
    # Min-max normalization
    X = torch.stack((D, D_grad, D_acc, D_peak, T1)).permute(1,0)
    minmax = embedding.get_max_min(X)
    X = embedding.norm(X, minmax)
    
    # Reservoir setup
    D_in_x = X.size(-1)
    W_in_x = embedding.map_in(D_in_x, Dx, Scale)
    W_res = embedding.map_res(Dx, Spar, Spec)
    
    # ESN embedding                                                                                                               
    X_plot = embedding.esn_act(X, W_res, W_in_x, Leaky, alpha, Scale).squeeze(0)
    X_full = torch.cat((X, X_plot), dim = 1)
    
    if n == 5:
        X_full_5 = X_full
        
    # Training dataset
    X_tr[n, :, :] = X_full[:Train_len, :]
    Y_tr[n, :, :] = F[:Train_len].unsqueeze(1)
    
    # Testing dataset
    X_te[n, :, :] = X_full[Train_len:, :]
    Y_te[n, :, :] = F[Train_len:].unsqueeze(1)

    print('Task {:1d} is ready'.format(n + 1))


# Dividing batch
X_in = torch.zeros(8, X_tr.size(1) - 2, 3, X_tr.size(2))
Y_in = torch.zeros(8, X_tr.size(1) - 2, 3, 1)

for n in range(8):
    for i in range(X_in.size(1)):
        X_in[n, i, :, :] = X_tr[n, i: i + 3, :]
        Y_in[n, i, :, 0] = Y_tr[n, i: i + 3, 0]

torch.save(X_in, 'Reptile/X_in.pth')
torch.save(Y_in, 'Reptile/Y_in.pth')
torch.save(X_te, 'Reptile/X_te.pth')
torch.save(Y_te, 'Reptile/Y_te.pth')
torch.save(X_full_5, 'Reptile/X_full_5.pth')
torch.save(Y_full_5, 'Reptile/Y_full_5.pth')