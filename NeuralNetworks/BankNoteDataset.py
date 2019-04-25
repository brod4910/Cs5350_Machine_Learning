import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class BankNoteDataset(Dataset):
    def __init__(self, csv_file, in_features):
        self.csv_data = pd.read_csv(csv_file)
        self.in_features = in_features

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        data = self.csv_data.iloc[idx, 0:self.in_features]

        label = self.csv_data.iloc[idx, -1]

        return {'input' : np.array(data, dtype=np.float32), 'label' : np.array(label)}