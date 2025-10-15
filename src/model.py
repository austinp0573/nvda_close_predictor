"""neural network models for stock price prediction"""
import torch
import torch.nn as nn

class FeedforwardPredictor(nn.Module):
    """simple feedforward neural network for stock price prediction"""
    
    def __init__(self, input_size=22, hidden_sizes=[64, 32, 16], dropout=0.3):
        """
        args:
            input_size: number of input features
            hidden_sizes: list of hidden layer sizes
            dropout: dropout probability
        """
        super(FeedforwardPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class LSTMPredictor(nn.Module):
    """lstm neural network for sequence-based stock price prediction"""
    
    def __init__(self, input_size=22, hidden_size=32, num_layers=2, dropout=0.3):
        """
        args:
            input_size: number of input features
            hidden_size: number of lstm hidden units
            num_layers: number of lstm layers
            dropout: dropout probability
        """
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # lstm layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # fully connected layers
        self.fc1 = nn.Linear(hidden_size, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # lstm forward pass
        lstm_out, _ = self.lstm(x)
        
        # take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # fully connected layers
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out


def count_parameters(model):
    """count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)