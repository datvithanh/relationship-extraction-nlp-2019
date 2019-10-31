import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch
from utils import load_data_and_labels

class TrainingDataset(Dataset):
    def __init__(self, file_path):
        self.tokens, self.labels, self.e1, self.e2, self.pos1, self.pos2 = load_data_and_labels(file_path)

    def __getitem__(self, index):
        return self.tokens[index], self.labels[index], self.e1[index], self.e2[index], self.pos1[index], self.pos2[index]

    def __len__(self):
        return len(self.tokens)

def LoadDataset(split, data_path, n_jobs=8, train_set='train', dev_set='dev', batch_size=8, cuda = True):
    if split == 'train':
        shuffle = True
        dataset_file = train_set + '.csv'
    else:
        shuffle = False
        dataset_file = dev_set + '.csv'

    ds = TrainingDataset(os.path.join(data_path, dataset_file))

    return  DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=n_jobs, pin_memory=cuda)