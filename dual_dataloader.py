from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import pytorch_lightning as pl
import torch.nn as nn
from model.models import Channel_Alignment
import os
import numpy as np
from data.dataset import EEG_Dataset, SEEDV_Dataset_new, FACED_Dataset_new, PretrainSampler
from torch.utils.data import Dataset, DataLoader

def get_train_subs(n_subs, fold, n_folds):
    n_per = round(n_subs / n_folds)
    if n_folds == 1:
        val_subs = []
    elif fold < n_folds - 1:
        val_subs = np.arange(n_per * fold, n_per * (fold + 1))
    else:
        val_subs = np.arange(n_per * fold, n_subs)            
    train_subs = list(set(np.arange(n_subs)) - set(val_subs))
    if len(val_subs) == 1:
        val_subs = list(val_subs) + train_subs
    return train_subs, val_subs
class PairedDataset(Dataset):
    def __init__(self, dataset_a, dataset_b):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.len_a = len(self.dataset_a)
        self.len_b = len(self.dataset_b)

    def __len__(self):
        return max(len(self.dataset_a), len(self.dataset_b))

    def __getitem__(self, idx):
        item_a = self.dataset_a[idx % self.len_a]
        item_b = self.dataset_b[idx % self.len_b]
        return item_a, item_b

class MultiEEGDataModule(pl.LightningDataModule):
    def __init__(self, data1, data2, fold, n_folds, num_workers=8):
        super().__init__()
        self.data1 = data1
        self.data2 = data2
        self.dataset_dual = None
        self.train_subs_1, self.val_subs_1 = get_train_subs(data1.n_subs, fold, n_folds)
        self.train_subs_2, self.val_subs_2 = get_train_subs(data2.n_subs, fold, n_folds)
        print(f'Dataset 1: {self.data1.dataset_name}\ntrain_subs:{self.train_subs_1}\nval_sbs:{self.val_subs_1}')
        print(f'Dataset 2: {self.data2.dataset_name}\ntrain_subs:{self.train_subs_2}\nval_sbs:{self.val_subs_2}')
    def prepare_data(self) -> None:
        pass
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset_1 = EEG_Dataset(self.data1, train_subs=self.train_subs_1, mods='train',sliced=False)
            self.trainset_2 = EEG_Dataset(self.data2, train_subs=self.train_subs_2, mods='train',sliced=False)
            self.train_dataset = PairedDataset(self.trainset_1, self.trainset_2)

            self.valset_1 = EEG_Dataset(self.data1, val_subs=self.val_subs_1, mods='val',sliced=False)
            self.valset_2 = EEG_Dataset(self.data2, val_subs=self.val_subs_2, mods='val',sliced=False)
            self.val_dataset = PairedDataset(self.valset_1, self.valset_2)
        if stage == 'test':
            self.valset_1 = EEG_Dataset(self.data1, val_subs=self.val_subs_1, mods='val',sliced=False)
            self.valset_2 = EEG_Dataset(self.data2, val_subs=self.val_subs_2, mods='val',sliced=False)
            self.val_dataset = PairedDataset(self.valset_1, self.valset_2)
    def train_dataloader(self) -> torch.Any:
        return DataLoader(self.train_dataset, batch_size=64, shuffle=True, num_workers=64)
    def val_dataloader(self) -> torch.Any:
        return DataLoader(self.val_dataset, batch_size=64, num_workers=64)
        
    



