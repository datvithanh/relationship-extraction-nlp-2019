#dataset
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np
from utils import load_data_and_labels

class TrainingDataset(Dataset):
    def __init__(self, file_path):
        tokens, self.labels, self.e1, self.e2, self.pos1, self.pos2 = load_data_and_labels(file_path)
        self.embedding = torch.load('train_embedding.ts')
        # print(self.embedding.shape, len(self.labels), len(self.e1), len(self.e2), len(self.pos1), len(self.pos2))
        # self.embedding = torch.zeros(len(tokens), 90, 1024)
        # for i in range(len(tokens)):
        #     # print(len(tokens[i]))
        #     # print(elmo.embed_sentence(tokens[i]).shape)
        #     self.embedding[i, :len(tokens[i]), :] = torch.from_numpy(elmo.embed_sentence(tokens[i])[2])

    def __getitem__(self, index):
        return self.embedding[index], self.labels[index], self.e1[index], self.e2[index], self.pos1[index], self.pos2[index]

    def __len__(self):
        return len(self.labels)

def LoadDataset(split, data_path, n_jobs=8, batch_size=8):
    print(f'Loading {split} data from {data_path}')
    if split == 'train':
        shuffle = True
    else:
        shuffle = False

    ds = TrainingDataset(os.path.join(data_path))

    return  DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=n_jobs)