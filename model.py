import torch
from torch import nn

class ESN(nn.Module):
    def __init__(self, d, d_out = 1):
        super(ESN, self).__init__()
        self.linear1 = nn.Linear(d, 25, bias=True)
        self.linear2 = nn.Linear(25,25, bias=True)
        self.linear3 = nn.Linear(25, d_out, bias=True)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class vESN(nn.Module):
    def __init__(self, d, d_out = 1):
        super(vESN, self).__init__()
        self.linear1 = nn.Linear(d, d_out, bias=True)

    def forward(self, feature):
        feature = self.linear1(feature)
        return feature


class LSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=1, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, 
                           hidden_size=hidden_dim, 
                           num_layers=num_layers, 
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.shape[1] <= 1000:  
            x = x.contiguous()
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out)
            return output
        else:
            chunk_size = 1000
            outputs = []
            
            h_t = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
            c_t = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
            
            for i in range(0, x.size(1), chunk_size):
                chunk = x[:, i:i+chunk_size, :]
                chunk = chunk.contiguous()
                
                lstm_out, (h_t, c_t) = self.lstm(chunk, (h_t, c_t))
                chunk_output = self.fc(lstm_out)
                outputs.append(chunk_output)
            
            return torch.cat(outputs, dim=1)
        
        
class RNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=1, num_layers=1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim,
                         hidden_size=hidden_dim,
                         num_layers=num_layers,
                         batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.shape[1] <= 1000:
            x = x.contiguous()
            rnn_out, _ = self.rnn(x)
            output = self.fc(rnn_out)
            return output
        else:
            chunk_size = 1000
            outputs = []
            
            h_t = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
            
            for i in range(0, x.size(1), chunk_size):
                chunk = x[:, i:i+chunk_size, :]
                chunk = chunk.contiguous()
                
                rnn_out, h_t = self.rnn(chunk, h_t)
                chunk_output = self.fc(rnn_out)
                outputs.append(chunk_output)
            
            return torch.cat(outputs, dim=1)


class Transformer(nn.Module):
    def __init__(self, input_dim=5, d_model=16, nhead=2, num_layers=1, output_dim=1):
        super(Transformer, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_proj = nn.Linear(d_model, output_dim)
        
        self.d_model = d_model

    def forward(self, x):
        if x.shape[1] <= 1000:
            # Project input to d_model dimensions
            x = self.input_proj(x)
            
            # Transformer expects input of shape (batch, seq_len, d_model)
            output = self.transformer_encoder(x)
            output = self.output_proj(output)
            return output
        else:
            chunk_size = 1000
            outputs = []
            
            for i in range(0, x.size(1), chunk_size):
                chunk = x[:, i:i+chunk_size, :]
                chunk = chunk.contiguous()
                
                # Process chunk
                chunk = self.input_proj(chunk)
                chunk_out = self.transformer_encoder(chunk)
                chunk_output = self.output_proj(chunk_out)
                outputs.append(chunk_output)
            
            return torch.cat(outputs, dim=1)