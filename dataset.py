#dataset
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np
from utils import load_data_and_labels

class TrainingDataset(Dataset):
    def __init__(self, data):
        self.embedding, _, self.labels, self.e1, self.e2, self.pos1, self.pos2 = zip(*data)

    def __getitem__(self, index):
        return self.embedding[index], self.labels[index], self.e1[index], self.e2[index], self.pos1[index], self.pos2[index]

    def __len__(self):
        return len(self.labels)

def LoadDataset(split, dataset, n_jobs=8, batch_size=16):
    if split == 'train':
        shuffle = True
    else:
        shuffle = False

    ds = TrainingDataset(dataset)

    return  DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=n_jobs)