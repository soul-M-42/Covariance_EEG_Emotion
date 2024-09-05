from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import pytorch_lightning as pl
import torch.nn as nn
from model.models import Channel_Alignment
import os
import numpy as np
from data.dataset import EEG_Dataset, SEEDV_Dataset_new, FACED_Dataset_new
from torch.utils.data import Dataset, DataLoader
import random
from itertools import cycle


class EEGSampler:
    def __init__(self, set1, set2, n_pairs=256):
        self.n_pairs = n_pairs
        self.set1 = set1
        self.set2 = set2
        self.subs1 = self.set1.train_subs if self.set1.train_subs is not None else self.set1.val_subs
        self.subs2 = self.set2.train_subs if self.set2.train_subs is not None else self.set2.val_subs
        self.n_sub1 = len(self.subs1)
        self.n_sub2 = len(self.subs2)
        self.n_session1 = self.set1.n_session
        self.n_session2 = self.set2.n_session
    
    def get_sample(self, set, subsession):
        save_dir = os.path.join(set.cfg.data_dir,'sliced_data')
        sliced_data_dir = os.path.join(save_dir, f'sliced_len{set.cfg.timeLen}_step{set.cfg.timeStep}')
        n_samples_session = np.load(sliced_data_dir+'/metadata/n_samples_sessions.npy')
        n_vid = set.cfg.n_vids
        batch_size = n_vid
        n_session = set.cfg.n_session
        cur_session = int(subsession % n_session)
        cur_sub = int(subsession // n_session)
        n_per_session = np.sum(n_samples_session,1).astype(int)
        n_per_session_cum = np.concatenate((np.array([0]), np.cumsum(n_per_session)))
        n_samples_per_trial = int(n_vid / n_samples_session.shape[1])
        n_samples_cum_session = np.concatenate((np.zeros((n_session,1)), np.cumsum(n_samples_session,1)),1)

        ind_abs = np.zeros(0)
        for i in range(len(n_samples_cum_session[cur_session])-2):
            ind_one = np.random.choice(np.arange(n_samples_cum_session[cur_session][i], n_samples_cum_session[cur_session][i+1]),
                                        n_samples_per_trial, replace=False)
            ind_abs = np.concatenate((ind_abs, ind_one))

        i = len(n_samples_cum_session[cur_session]) - 2
        ind_one = np.random.choice(np.arange(n_samples_cum_session[cur_session][i], n_samples_cum_session[cur_session][i + 1]),
                                    int(batch_size - len(ind_abs)), replace=False)
        ind_abs = np.concatenate((ind_abs, ind_one))
            # print('ind abs length', len(ind_abs))

        ind_this = ind_abs + np.sum(n_per_session)*cur_sub + n_per_session_cum[cur_session]
        return ind_this

    def __len__(self):
        return self.n_pairs
        n_pair_1 = self.n_sub1*self.n_session1
        n_pair_2 = self.n_sub2*self.n_session2
        return n_pair_1 * n_pair_2

    def __iter__(self):
        pairs_1 = []
        pairs_2 = []
        for i in range(self.n_sub1*self.n_session1):
            for j in range(i+1, self.n_sub1*self.n_session1):
                if(int(i % self.n_session1) == int(j % self.n_session1)):
                    # from the same session
                    pairs_1.append((i,j))
        for i in range(self.n_sub2*self.n_session2):
            for j in range(i+1, self.n_sub2*self.n_session2):
                if(int(i % self.n_session2) == int(j % self.n_session2)):
                    # from the same session
                    pairs_2.append((i,j))
        
        # for pair1 in pairs_1:
        #     for pair2 in pairs_2:
        #         idx_1 = np.array([])
        #         idx_2 = np.array([])
        #         idx_1 = np.concatenate((self.get_sample(set=self.set1, subsession=pair1[0]),
        #                                 self.get_sample(set=self.set1, subsession=pair1[1])))
        #         idx_2 = np.concatenate((self.get_sample(set=self.set2, subsession=pair2[0]),
        #                                 self.get_sample(set=self.set2, subsession=pair2[1])))
        #         # idx_1 = np.random.randint(0, len(self.set1), size=self.batch_size)  # Index from dataset 1
        #         # idx_2 = np.random.randint(0, len(self.set2), size=self.batch_size)  # Index from dataset 2
                
        #         # Yield a tuple containing both indices
        #         yield list(idx_1.astype(int)), list(idx_2.astype(int))

        for i in range(self.n_pairs):
            pair1 = random.choice(pairs_1)
            pair2 = random.choice(pairs_2)
            idx_1 = np.array([])
            idx_2 = np.array([])
            idx_1 = np.concatenate((self.get_sample(set=self.set1, subsession=pair1[0]),
                                    self.get_sample(set=self.set1, subsession=pair1[1])))
            idx_2 = np.concatenate((self.get_sample(set=self.set2, subsession=pair2[0]),
                                    self.get_sample(set=self.set2, subsession=pair2[1])))
            # idx_1 = np.random.randint(0, len(self.set1), size=self.batch_size)  # Index from dataset 1
            # idx_2 = np.random.randint(0, len(self.set2), size=self.batch_size)  # Index from dataset 2
            
            # Yield a tuple containing both indices
            yield list(idx_1.astype(int)), list(idx_2.astype(int))


        


class ZipLongestRepeat:
    def __init__(self, *iterables):
        # Store the iterables and calculate the maximum length
        self.iterables = iterables
        self.max_length = min(len(it) for it in iterables)
        self.iterators = [cycle(it) for it in iterables]
        self.current_iteration = 0
    
    def __len__(self):   #n_batch
        return self.max_length

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_iteration >= self.max_length:
            raise StopIteration
        
        values = []
        for iterator in self.iterators:
            value = next(iterator)
            values.append(value)
        
        self.current_iteration += 1
        return tuple(values)

def get_train_subs(n_subs, fold, n_folds):
    n_per = round(n_subs / n_folds)
    if n_folds == 1:
        val_subs = []
    elif fold < n_folds - 1:
        val_subs = np.arange(n_per * fold, n_per * (fold + 1))
    else:
        val_subs = np.arange(n_per * fold, n_subs)            
    train_subs = list(set(np.arange(n_subs)) - set(val_subs))
    # if len(val_subs) == 1:
    #     val_subs = list(val_subs) + train_subs
    return train_subs, val_subs
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
        data_a = []
        label_a = []
        data_b = []
        label_b = []
        for idx in idx_a:
            data_a.append(self.dataset_a[idx][0])
            label_a.append(self.dataset_a[idx][1])
        for idx in idx_b:
            data_b.append(self.dataset_b[idx][0])
            label_b.append(self.dataset_b[idx][1])
        data_a = torch.stack(data_a)
        data_b = torch.stack(data_b)
        label_a = torch.stack(label_a)
        label_b = torch.stack(label_b)
        return data_a, label_a, data_b, label_b

# class MultiEEGDataModule(pl.LightningDataModule):
#     def __init__(self, data1, data2, fold, n_folds, batch_size=64, num_workers=8, device='cpu'):
#         super().__init__()
#         self.device = device
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.data1 = data1
#         self.data2 = data2
#         self.dataset_dual = None
#         self.train_subs_1, self.val_subs_1 = get_train_subs(data1.n_subs, fold, n_folds)
#         self.train_subs_2, self.val_subs_2 = get_train_subs(data2.n_subs, fold, n_folds)
#         print(f'Dataset 1: {self.data1.dataset_name}\ntrain_subs:{self.train_subs_1}\nval_sbs:{self.val_subs_1}')
#         print(f'Dataset 2: {self.data2.dataset_name}\ntrain_subs:{self.train_subs_2}\nval_sbs:{self.val_subs_2}')
#     def prepare_data(self) -> None:
#         pass
#     def setup(self, stage=None):
#         self.save_dir_1 = os.path.join(self.data1.data_dir,'sliced_data')
#         self.sliced_data_dir_1 = os.path.join(self.save_dir_1, f'sliced_len{self.data1.timeLen}_step{self.data1.timeStep}')
#         self.n_samples_sessions_1 = np.load(self.sliced_data_dir_1+'/metadata/n_samples_sessions.npy')
#         self.save_dir_2 = os.path.join(self.data2.data_dir,'sliced_data')
#         self.sliced_data_dir_2 = os.path.join(self.save_dir_2, f'sliced_len{self.data2.timeLen}_step{self.data2.timeStep}')
#         self.n_samples_sessions_2 = np.load(self.sliced_data_dir_2+'/metadata/n_samples_sessions.npy')

#         if stage == 'fit' or stage is None:
#             self.trainset_1 = EEG_Dataset(self.data1, train_subs=self.train_subs_1, mods='train',sliced=False)
#             self.trainset_2 = EEG_Dataset(self.data2, train_subs=self.train_subs_2, mods='train',sliced=False)
#             self.valset_1 = EEG_Dataset(self.data1, val_subs=self.val_subs_1, mods='val',sliced=False)
#             self.valset_2 = EEG_Dataset(self.data2, val_subs=self.val_subs_2, mods='val',sliced=False)


#         if stage == 'validate':
#             self.valset_1 = EEG_Dataset(self.data1, val_subs=self.val_subs_1, mods='val',sliced=False)
#             self.valset_2 = EEG_Dataset(self.data2, val_subs=self.val_subs_2, mods='val',sliced=False)

#     def train_dataloader(self) -> torch.Any:
#         self.trainsampler1 = EEGSampler(n_subs=len(self.train_subs_1), batch_size=self.data1.n_vids,
#                                         n_samples_session=self.n_samples_sessions_1, n_times=1)
#         self.trainloader_1 = DataLoader(self.trainset_1, batch_sampler=self.trainsampler1, pin_memory=True, num_workers=self.num_workers, generator=torch.Generator(device='cuda'))
#         self.trainsampler2 = EEGSampler(n_subs=len(self.train_subs_2), batch_size=self.data2.n_vids,
#                                         n_samples_session=self.n_samples_sessions_2, n_times=1)
#         self.trainloader_2 = DataLoader(self.trainset_2, batch_sampler=self.trainsampler2, pin_memory=True, num_workers=self.num_workers, generator=torch.Generator(device='cuda'))
        
#         self.trainsampler = EEGSampler(set1=self.trainset_1, set2=self.trainset_2)
#         self.trainloader = DataLoader(datas)
#         return self.trainloader_1
#         return ZipLongestRepeat(self.trainloader_1, self.trainloader_2)
   
#     def val_dataloader(self) -> torch.Any:
#         self.valsampler1 = EEGSampler(n_subs=len(self.val_subs_1), batch_size=self.data1.n_vids,
#                                             n_samples_session=self.n_samples_sessions_1, n_times=1)
#         self.valloader_1 = DataLoader(self.valset_1, batch_sampler=self.valsampler1, pin_memory=True, num_workers=self.num_workers, generator=torch.Generator(device='cuda'))
#         self.valsampler2 = EEGSampler(n_subs=len(self.val_subs_2), batch_size=self.data2.n_vids,
#                                         n_samples_session=self.n_samples_sessions_2, n_times=1)
#         self.valloader_2 = DataLoader(self.valset_2, batch_sampler=self.valsampler2, pin_memory=True, num_workers=self.num_workers, generator=torch.Generator(device='cuda'))
#         return self.valloader_1
#         return ZipLongestRepeat(self.valloader_1, self.valloader_2)

class DualDataModule(pl.LightningDataModule):
    def __init__(self, data1, data2, fold, n_folds, batch_size=64, n_pairs=128, num_workers=8, device='cpu'):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
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
        self.save_dir_1 = os.path.join(self.data1.data_dir,'sliced_data')
        self.sliced_data_dir_1 = os.path.join(self.save_dir_1, f'sliced_len{self.data1.timeLen}_step{self.data1.timeStep}')
        self.n_samples_sessions_1 = np.load(self.sliced_data_dir_1+'/metadata/n_samples_sessions.npy')
        self.save_dir_2 = os.path.join(self.data2.data_dir,'sliced_data')
        self.sliced_data_dir_2 = os.path.join(self.save_dir_2, f'sliced_len{self.data2.timeLen}_step{self.data2.timeStep}')
        self.n_samples_sessions_2 = np.load(self.sliced_data_dir_2+'/metadata/n_samples_sessions.npy')

        if stage == 'fit' or stage is None:
            self.trainset_1 = EEG_Dataset(self.data1, train_subs=self.train_subs_1, mods='train',sliced=False)
            self.trainset_2 = EEG_Dataset(self.data2, train_subs=self.train_subs_2, mods='train',sliced=False)
            self.valset_1 = EEG_Dataset(self.data1, val_subs=self.val_subs_1, mods='val',sliced=False)
            self.valset_2 = EEG_Dataset(self.data2, val_subs=self.val_subs_2, mods='val',sliced=False)
            self.trainset = PairedDataset(self.trainset_1, self.trainset_2)
            self.valset = PairedDataset(self.valset_1, self.valset_2)


        if stage == 'validate':
            self.valset_1 = EEG_Dataset(self.data1, val_subs=self.val_subs_1, mods='val',sliced=False)
            self.valset_2 = EEG_Dataset(self.data2, val_subs=self.val_subs_2, mods='val',sliced=False)
            self.valset = PairedDataset(self.valset_1, self.valset_2)

    def train_dataloader(self) -> torch.Any:
        self.trainsampler = EEGSampler(set1=self.trainset_1, set2=self.trainset_2, n_pairs=self.n_pairs)
        self.trainloader = DataLoader(self.trainset, sampler=self.trainsampler, pin_memory=True, num_workers=self.num_workers, generator=torch.Generator(device='cuda'))
        return self.trainloader
   
    def val_dataloader(self) -> torch.Any:
        self.valsampler = EEGSampler(set1=self.valset_1, set2=self.valset_2, n_pairs=self.n_pairs)
        self.valloader = DataLoader(self.valset, sampler=self.valsampler, pin_memory=True, num_workers=self.num_workers, generator=torch.Generator(device='cuda'))
        return self.valloader
    



