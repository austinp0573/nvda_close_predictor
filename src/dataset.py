"""pytorch dataset classes for nvda stock price prediction"""
import torch
from torch.utils.data import Dataset
import numpy as np

class NVDADataset(Dataset):
    """pytorch dataset for nvda stock price prediction"""
    
    def __init__(self, X, y):
        """
        args:
            X: numpy array of shape (n, features)
            y: numpy array of shape (n, 1) or (n,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NVDASequenceDataset(Dataset):
    """pytorch dataset for lstm with sequences"""
    
    def __init__(self, X, y, lookback=20):
        """
        args:
            X: numpy array of shape (n, features)
            y: numpy array of shape (n, 1) or (n,)
            lookback: number of time steps to look back
        """
        self.lookback = lookback
        self.X_sequences = []
        self.y_sequences = []
        
        # create sequences
        for i in range(len(X) - lookback):
            self.X_sequences.append(X[i:i+lookback])
            self.y_sequences.append(y[i+lookback])
        
        self.X_sequences = torch.FloatTensor(np.array(self.X_sequences))
        self.y_sequences = torch.FloatTensor(np.array(self.y_sequences))
        
    def __len__(self):
        return len(self.X_sequences)
    
    def __getitem__(self, idx):
        return self.X_sequences[idx], self.y_sequences[idx]