import torch

def spike(data):
    t = data.size(0)
    data_s = torch.zeros(t, dtype=torch.float, requires_grad=False)
    for i in range(1, t - 1):
        if torch.abs(data[i]) > torch.abs(data[i - 1]) and torch.abs(data[i]) > torch.abs(data[i + 1]):
            data_s[i] = data[i]
    return data_s


def grad(data):
    t = data.size(0)
    data_g = torch.zeros(t, dtype=torch.float, requires_grad=False)
    for i in range(1, t - 1):
        data_g[i] = data[i + 1] - data[i - 1]
    data_g[0] = data[1] - data[0]
    data_g[-1] = data[-1] - data[-2]
    return data_g


def decay(data, log, power):
    t = data.size(0)
    data_t1 = torch.log(0.01 * torch.arange(log, t + log, dtype=torch.float, requires_grad=False)) * -1
    data_t1 = (data_t1 + -1 * torch.min(data_t1)) / (-1 * torch.min(data_t1))
    data_t2 = power ** torch.arange(1, t + 1, dtype=torch.float, requires_grad=False) * -1
    return data_t1, data_t2


def get_max_min(tensor):
    d = tensor.size(1)
    result_tensor = torch.zeros(2, d)
    for i in range(d):
        max_val = torch.max(tensor[:, i])
        min_val = torch.min(tensor[:, i])
    
        result_tensor[0, i] = max_val
        result_tensor[1, i] = min_val
    return result_tensor


def norm(x, minmax):
    # norm range (0, 1)
    num_b = 1
    d = x.size(1)
    width = (minmax[0,:] - minmax[1,:])
    for i in range(num_b):
        for j in range(d):
            x[:, j] = (x[:, j] - minmax[1, j]) / width[j]
    return x


def extract_sequence(S):
    result = []
    start = 0
    while start < S.size(0):
        end = start + 5
        result.append(S[start:end])
        start = end + 5
    return torch.cat(result)


def map_res(d, spar, spec):
    w_res = torch.diag(torch.randn(d, dtype=torch.float, requires_grad=False))
    w_res = w_res[torch.randperm(d)]
    if spar - 1 > 0:
        num_sup = int(d * (spar - 1))
        for i in range(num_sup):
            sup = torch.randn(1, dtype=torch.float, requires_grad=False)
            while True:
                raw = torch.randint(0, d, [1])
                col = torch.randint(0, d, [1])
                if w_res[raw[0], col[0]] == 0:
                    break
            w_res[raw[0], col[0]] = sup[0]
    eigen_v = torch.linalg.eigvals(w_res)
    spec_r = max(abs(eigen_v))
    w_res = spec * w_res / spec_r
    return w_res


def map_in(d_in, d, scale):
    w_in = torch.rand((d, d_in + 1), dtype=torch.float, requires_grad=False)
    for i in range(d):
        for j in range(d_in):
            w_in[i, j] = w_in[i, j] / torch.sum(w_in[i])
    w_in = scale * w_in
    return w_in


def esn_act(x, w_res, w_in, Leaky_r, alpha, scale):
    num_res = w_res.size(0)
    num_b = 1
    t = x.size(0)
    feature = torch.zeros((num_b, t, num_res), dtype=torch.float)
    for i in range(num_b):
        x_f = x
        x_b = torch.flip(x_f, [0])
        # Bi-directional states
        h_f = torch.zeros((t + 1, num_res), dtype=torch.float)
        h_b = torch.zeros((t + 1, num_res), dtype=torch.float) 
        bias_x = torch.tensor([scale], dtype=torch.float)
        for j in range(t):
            h_f[j + 1] = (1 - Leaky_r) * h_f[j] + Leaky_r * torch.tanh(w_res @ h_f[j] 
                                                                     + w_in @ torch.cat((x_f[j], bias_x), 0))
            h_b[j + 1] = (1 - Leaky_r) * h_b[j] + Leaky_r * torch.tanh(w_res @ h_b[j] 
                                                                     + w_in @ torch.cat((x_b[j], bias_x), 0))
        feature[i] = h_f[1:]
# =============================================================================
#         feature[i] = torch.cat((h_f[1:], 
#                                 torch.flip(h_b[1:], [0])), 1)
# =============================================================================
          
    return feature












