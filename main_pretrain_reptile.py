import torch
import random
import model
import time
import os
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import dataloaders
# =============================================================================
X_in = torch.load('Reptile/X_in_entire.pth', weights_only=False)
Y_in = torch.load('Reptile/Y_in_entire.pth', weights_only=False)
X_te = torch.load('Reptile/X_te_entire.pth', weights_only=False)
Y_te = torch.load('Reptile/Y_te_entire.pth', weights_only=False)
print('Loading completed')
# =============================================================================

# Inner loop learning rate
inner_lr = 0.0001
# meta learning rate
meta_lr = 0.2
# Number outer loop
Num_out = 3
# Adam decay
Lamda = 0.001
# Inner epoch
inner_epoch = 50
# Batch size
Num_bat = 10
# Model dimension
Dimension = X_in.size(3)

# Establish Tensor dataset loader
Dataset_list = []
Taskloader_list = []
for n in range(8):
    task_data = TensorDataset(X_in[n,:,:,:].detach(), Y_in[n,:,:,:].detach())
    task_loader = DataLoader(task_data, batch_size=Num_bat, shuffle=True)
    Dataset_list.append(task_data)
    Taskloader_list.append(task_loader)

# Gradient function
def grad(Y):
    Y_grad = torch.stack(((Y[:, 1, 0] - Y[:, 0, 0]),
                          (Y[:, 2, 0] - Y[:, 1, 0])))
    return Y_grad

# Inner training loop
def reptile_update(Model, task_loader, inner_optimizer, inner_epoch, loss_fn, meta_lr):
    orig_params = {name: param.data.clone() for name, param in Model.named_parameters()}
    loss_log = torch.zeros(inner_epoch, dtype=torch.float)
    # Inner epoch
    for i in range(inner_epoch):
        epoch_start_time = time.time()
        # Trainig batch
        for X, Y in task_loader:
            # Output true value
            Y_hat = Model(X.to(device))
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
            loss_all = loss + 5 * loss_p + 0.5 * loss_g
            
            # Optimization
            inner_optimizer.zero_grad()
            loss_all.backward()
            inner_optimizer.step()
            
        loss_log[i] = loss.detach()
        # Monitor training log   
        if i % 10 == 9:            
            # Inner loop evalucation
            print('| end of epoch {:3d} | time: {:5.2f}s | Train loss {:5.5f} |'.
                  format(i + 1 , (time.time() - epoch_start_time),
                         1e5 * loss.to('cpu').detach() ))
    # Reptile update            
    with torch.no_grad():
        for name, param in Model.named_parameters():
            if param.requires_grad:
                param.data = orig_params[name] + meta_lr * (param.data - orig_params[name])
    return loss_log
            
# Model setup
Model = model.ESN(Dimension).to(device)
loss_fn = nn.MSELoss()
inner_optimizer = torch.optim.Adam(Model.parameters(), lr=inner_lr)

# Outer training loop
target_task = 0 # one target in range(7)
random_ints = []

# Selecting outer loop idx
target = 0 
sequence = list(range(8))
loop_idx = sequence[:]
loop_idx.remove(target)
loop_idx = loop_idx * Num_out
random.shuffle(loop_idx)

# Outer epoch
outer_epoch = len(loop_idx)
reptile_log = torch.zeros(outer_epoch, inner_epoch, dtype=torch.float)

# Outer loop updating
for m in range(outer_epoch):       
    # Sampling taskloader
    idx = loop_idx[m]
    
    print('\n' + "=" * 17 + 'Outer loop ' + str(m+1) + ' on task ' + str(idx+1) + "=" * 17)
    task_loader = Taskloader_list[idx]
    reptile_log[m, :] = reptile_update(Model, task_loader, inner_optimizer, inner_epoch, loss_fn, meta_lr)

torch.save(Model, 'Reptile/Pre_Model-entire.pkl')
torch.save(reptile_log, 'Reptile/Pretrain_log-entire.pth')































