from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import pytorch_lightning as pl
import torch.nn as nn
import os
import numpy as np
from data.dataset import EEG_Dataset, SEEDV_Dataset_new, FACED_Dataset_new
from torch.utils.data import Dataset, DataLoader
import random
from itertools import cycle


def get_train_subs(n_subs, fold, n_folds):
    n_per = round(n_subs / n_folds)
    if n_folds == 1:
        val_subs = []
    elif fold < n_folds - 1:
        val_subs = np.arange(n_per * fold, n_per * (fold + 1))
    else:
        val_subs = np.arange(n_per * fold, n_subs)
    train_subs = list(set(np.arange(n_subs)) - set(val_subs))
    return train_subs, val_subs


class EEGSampler:
    def __init__(self, set1, set2, n_pairs):
        self.n_pairs = n_pairs
        self.set1 = set1
        self.set2 = set2
        self.subs1 = self.set1.train_subs if self.set1.train_subs is not None else self.set1.val_subs
        self.subs2 = self.set2.train_subs if self.set2.train_subs is not None else self.set2.val_subs
        self.n_sub1 = len(self.subs1)
        self.n_sub2 = len(self.subs2)
        self.n_session1 = self.set1.n_session
        self.n_session2 = self.set2.n_session
        self.pairs_1 = []
        self.pairs_2 = []

        # Create pairs for set1
        for i in range(self.n_sub1 * self.n_session1):
            for j in range(i + self.n_session1, self.n_sub1 * self.n_session1, self.n_session1):
                if int(i % self.n_session1) == int(j % self.n_session1):
                    self.pairs_1.append((i, j))
        # Create pairs for set2
        for i in range(self.n_sub2 * self.n_session2):
            for j in range(i + self.n_session2, self.n_sub2 * self.n_session2, self.n_session2):
                if int(i % self.n_session2) == int(j % self.n_session2):
                    self.pairs_2.append((i, j))

        random.shuffle(self.pairs_1)
        random.shuffle(self.pairs_2)
        self.n_pairs_1 = len(self.pairs_1)
        self.n_pairs_2 = len(self.pairs_2)
        print('Dataloader Length:')
        print(self.n_pairs_1, self.n_pairs_2)
        self.max_n_pairs = max(self.n_pairs_1, self.n_pairs_2)

        # Preload data for set1
        self.save_dir_1 = os.path.join(self.set1.cfg.data_dir, 'sliced_data')
        self.sliced_data_dir_1 = os.path.join(
            self.save_dir_1, f'sliced_len{self.set1.cfg.timeLen}_step{self.set1.cfg.timeStep}')
        self.n_samples_session_1 = np.load(
            os.path.join(self.sliced_data_dir_1, 'metadata', 'n_samples_sessions.npy'))
        self.n_vid_1 = self.set1.cfg.n_vids
        self.batch_size_1 = self.n_vid_1
        self.n_session_1 = self.set1.cfg.n_session
        self.n_per_session_1 = np.sum(self.n_samples_session_1, 1).astype(int)
        self.n_per_session_cum_1 = np.concatenate((np.array([0]), np.cumsum(self.n_per_session_1)))
        self.n_samples_per_trial_1 = int(self.n_vid_1 / self.n_samples_session_1.shape[1])
        self.n_samples_cum_session_1 = np.concatenate(
            (np.zeros((self.n_session_1, 1)), np.cumsum(self.n_samples_session_1, 1)), 1)

        # Preload data for set2
        self.save_dir_2 = os.path.join(self.set2.cfg.data_dir, 'sliced_data')
        self.sliced_data_dir_2 = os.path.join(
            self.save_dir_2, f'sliced_len{self.set2.cfg.timeLen}_step{self.set2.cfg.timeStep}')
        self.n_samples_session_2 = np.load(
            os.path.join(self.sliced_data_dir_2, 'metadata', 'n_samples_sessions.npy'))
        self.n_vid_2 = self.set2.cfg.n_vids
        self.batch_size_2 = self.n_vid_2
        self.n_session_2 = self.set2.cfg.n_session
        self.n_per_session_2 = np.sum(self.n_samples_session_2, 1).astype(int)
        self.n_per_session_cum_2 = np.concatenate((np.array([0]), np.cumsum(self.n_per_session_2)))
        self.n_samples_per_trial_2 = int(self.n_vid_2 / self.n_samples_session_2.shape[1])
        self.n_samples_cum_session_2 = np.concatenate(
            (np.zeros((self.n_session_2, 1)), np.cumsum(self.n_samples_session_2, 1)), 1)

    def get_sample(self, set_idx, subsession_pair):
        if set_idx == 1:
            # Parameters for set1
            n_per_session = self.n_per_session_1
            n_per_session_cum = self.n_per_session_cum_1
            n_samples_cum_session = self.n_samples_cum_session_1
            n_session = self.n_session_1
            n_samples_per_trial = self.n_samples_per_trial_1
            n_sub = self.n_sub1
            batch_size = self.batch_size_1
        else:
            # Parameters for set2
            n_per_session = self.n_per_session_2
            n_per_session_cum = self.n_per_session_cum_2
            n_samples_cum_session = self.n_samples_cum_session_2
            n_session = self.n_session_2
            n_samples_per_trial = self.n_samples_per_trial_2
            n_sub = self.n_sub2
            batch_size = self.batch_size_2

        subsession1, subsession2 = subsession_pair

        # Ensure both subsessions are from the same session
        cur_session = int(subsession1 % n_session)
        assert cur_session == int(subsession2 % n_session), "Subsessions are from different sessions"

        cur_sub1 = int(subsession1 // n_session)
        cur_sub2 = int(subsession2 // n_session)

        ind_abs_list = []

        n_trials = len(n_samples_cum_session[cur_session]) - 2
        for i in range(n_trials):
            start = int(n_samples_cum_session[cur_session][i])
            end = int(n_samples_cum_session[cur_session][i + 1])
            ind_one = np.random.choice(np.arange(start, end), n_samples_per_trial, replace=False)
            ind_abs_list.append(ind_one)

        # For the remaining samples
        i = n_trials
        start = int(n_samples_cum_session[cur_session][i])
        end = int(n_samples_cum_session[cur_session][i + 1])
        remaining_samples = int(batch_size - n_samples_per_trial * n_trials)
        ind_one = np.random.choice(np.arange(start, end), remaining_samples, replace=False)
        ind_abs_list.append(ind_one)

        ind_abs = np.concatenate(ind_abs_list)

        # Compute indices for both subsessions
        ind_this1 = ind_abs + np.sum(n_per_session) * cur_sub1 + n_per_session_cum[cur_session]
        ind_this2 = ind_abs + np.sum(n_per_session) * cur_sub2 + n_per_session_cum[cur_session]

        return ind_this1, ind_this2

    def __len__(self):
        return self.n_pairs

    def __iter__(self):
        for i in range(self.n_pairs):
            index = random.randint(0, self.max_n_pairs - 1)
            pair1 = self.pairs_1[index % self.n_pairs_1]
            pair2 = self.pairs_2[index % self.n_pairs_2]

            # Get samples for set1
            idx_1_1, idx_1_2 = self.get_sample(set_idx=1, subsession_pair=pair1)
            idx_1 = np.concatenate((idx_1_1, idx_1_2))

            # Get samples for set2
            idx_2_1, idx_2_2 = self.get_sample(set_idx=2, subsession_pair=pair2)
            idx_2 = np.concatenate((idx_2_1, idx_2_2))

            yield list(idx_1.astype(int)), list(idx_2.astype(int))


class PairedDataset(Dataset):
    def __init__(self, dataset_a, dataset_b):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.len_a = len(self.dataset_a)
        self.len_b = len(self.dataset_b)
        print(len(self.dataset_a), len(self.dataset_b))

    def __len__(self):
        return max(len(self.dataset_a), len(self.dataset_b))

    def __getitem__(self, idx):
        idx_a, idx_b = idx
        data_a = [self.dataset_a[i][0] for i in idx_a]
        label_a = [self.dataset_a[i][1] for i in idx_a]
        data_b = [self.dataset_b[i][0] for i in idx_b]
        label_b = [self.dataset_b[i][1] for i in idx_b]
        data_a = torch.stack(data_a)
        data_b = torch.stack(data_b)
        label_a = torch.stack(label_a)
        label_b = torch.stack(label_b)
        return data_a, label_a, data_b, label_b


class DualDataModule(pl.LightningDataModule):
    def __init__(self, data1, data2, fold, n_folds, n_pairs=256, num_workers=8, device='cpu'):
        super().__init__()
        self.device = device
        self.n_pairs = n_pairs
        self.num_workers = num_workers
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
            self.trainset_1 = EEG_Dataset(self.data1, train_subs=self.train_subs_1, mods='train', sliced=False)
            self.trainset_2 = EEG_Dataset(self.data2, train_subs=self.train_subs_2, mods='train', sliced=False)
            self.valset_1 = EEG_Dataset(self.data1, val_subs=self.val_subs_1, mods='val', sliced=False)
            self.valset_2 = EEG_Dataset(self.data2, val_subs=self.val_subs_2, mods='val', sliced=False)
            self.trainset = PairedDataset(self.trainset_1, self.trainset_2)
            self.valset = PairedDataset(self.valset_1, self.valset_2)

        if stage == 'validate':
            self.valset_1 = EEG_Dataset(self.data1, val_subs=self.val_subs_1, mods='val', sliced=False)
            self.valset_2 = EEG_Dataset(self.data2, val_subs=self.val_subs_2, mods='val', sliced=False)
            self.valset = PairedDataset(self.valset_1, self.valset_2)

    def train_dataloader(self) -> torch.Any:
        self.trainsampler = EEGSampler(set1=self.trainset_1, set2=self.trainset_2, n_pairs=self.n_pairs)
        self.trainloader = DataLoader(self.trainset, sampler=self.trainsampler,
                                      pin_memory=True, num_workers=self.num_workers)
        return self.trainloader

    def val_dataloader(self) -> torch.Any:
        self.valsampler = EEGSampler(set1=self.valset_1, set2=self.valset_2, n_pairs=int(self.n_pairs // 4))
        self.valloader = DataLoader(self.valset, sampler=self.valsampler,
                                    pin_memory=True, num_workers=self.num_workers)
        return self.valloader
