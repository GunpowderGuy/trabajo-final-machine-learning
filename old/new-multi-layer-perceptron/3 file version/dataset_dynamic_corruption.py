# dataset_dynamic_corruption.py

import torch
from torch.utils.data import Dataset

class CorruptingDataset(Dataset):
    def __init__(self, X, mask, y, corruption_prob=0.1):
        self.X = X
        self.mask = mask
        self.y = y
        self.corruption_prob = corruption_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        m = self.mask[idx].clone()
        y = self.y[idx]

        noise_mask = torch.rand_like(x) < self.corruption_prob
        rand_indices = torch.randint(0, len(self.X), (x.size(0),))
        random_vals = self.X[rand_indices, torch.arange(x.size(0))]
        x[noise_mask] = random_vals[noise_mask]
        m[noise_mask] = 1.0

        x_aug = torch.cat([x, m], dim=0)
        return x_aug, y

  
