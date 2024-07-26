import hydra
from omegaconf import DictConfig
import torch
from model import Conv_att_simple_new, ExtractorModel, AlignedExtractorModel, Channel_Alignment
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data.pl_datamodule import EEGDataModule,SEEDVDataModule, FACEDDataModule
import os
import glob
import logging
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import sys
log = logging.getLogger(__name__)

def list_npy_files(directory):
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    return npy_files

def load_npy_files_to_matrix(directory):
    npy_files = list_npy_files(directory)
    data_list = [np.load(os.path.join(directory, f)) for f in npy_files]
    data_matrix = np.stack(data_list, axis=0)
    return data_matrix

def calculate_covariance_matrices(npy_matrix):
    num_samples = npy_matrix.shape[0]
    num_features = npy_matrix.shape[1]
    cov_matrices = np.zeros((num_samples, num_features, num_features))
    for i in range(num_samples):
        cov_matrices[i] = np.cov(npy_matrix[i], rowvar=True)
    return cov_matrices

def plot_covariance_matrix(cov_matrix, file_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap='viridis')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.savefig(file_path)
    plt.close()

def frobenius_distance(matrix_a, matrix_b):
    matrix_a = np.array(matrix_a)
    matrix_b = np.array(matrix_b)
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Must have same shape")
    # 计算Frobenius距离
    distance = np.linalg.norm(matrix_a - matrix_b, 'fro')
    return distance
def get_centroid(cov_matrices, tol=1e-5, max_iter=100):
    centroid = np.mean(cov_matrices, axis=0)
    
    for i in range(max_iter):
        prev_centroid = centroid.copy()
        weights = np.zeros(len(cov_matrices))
        for j, cov_matrix in enumerate(cov_matrices):
            weights[j] = 1.0 / frobenius_distance(cov_matrix, centroid)
        weights /= np.sum(weights)
        centroid = np.zeros_like(centroid)
        for j, cov_matrix in enumerate(cov_matrices):
            centroid += weights[j] * cov_matrix
        # 检查是否达到终止条件
        total_dis = 0
        for j, cov_matrix in enumerate(cov_matrices):
            total_dis += frobenius_distance(cov_matrix, centroid)
        delta = frobenius_distance(centroid, prev_centroid)
        print(f'Ite {i} total_dis {total_dis / cov_matrices.shape[0]} delta {delta}')
        if frobenius_distance(centroid, prev_centroid) < tol:
            break
    return centroid
@hydra.main(config_path="cfgs", config_name="config_cov", version_base="1.3")
def compute_cen(cfg: DictConfig) -> None:
    print(sys.argv[0])
    pl.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    n_folds = 1
    n_per = round(cfg.data.n_subs / n_folds)

    for fold in range(0,n_folds):
        # load data
        print("fold:", fold)
        cp_dir = os.path.join(cfg.log.cp_dir, cfg.data.dataset_name, f'r{cfg.log.run}')
        os.makedirs(cp_dir, exist_ok=True)
        # split data
        if n_folds == 1:
            val_subs = []
        elif fold < n_folds - 1:
            val_subs = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_subs = np.arange(n_per * fold, cfg.data.n_subs)            
        train_subs = list(set(np.arange(cfg.data.n_subs)) - set(val_subs))
        if len(val_subs) == 1:
            val_subs = list(val_subs) + train_subs
        print('train_subs:', train_subs)
        print('val_subs:', val_subs)
        
        # load_dir = os.path.join(cfg.data.data_dir,'processed_data')
        # save_dir = os.path.join(cfg.data.data_dir,'sliced_data')
        if cfg.data.dataset_name == 'FACED':
            if cfg.data.n_class == 2:
                n_vids = 24
            elif cfg.data.n_class == 9:
                n_vids = 28
        else:
            n_vids = cfg.data.n_vids
        train_vids = np.arange(n_vids)
        val_vids = np.arange(n_vids)
        #     pass
        dm = EEGDataModule(cfg.data, train_subs, val_subs, train_vids, val_vids,
                           cfg.train.valid_method=='loo', cfg.train.num_workers)
        
        
        checkpoint =  os.path.join(cfg.log.cp_dir,cfg.data_source.dataset_name,f'r{cfg.log.run}',f'f{fold}*')
        checkpoint = glob.glob(checkpoint)[0]
        
        log.info('checkpoint load from: '+checkpoint)
        Extractor = ExtractorModel.load_from_checkpoint(checkpoint_path=checkpoint)
        print(Extractor)
        
        sliced_dir = os.path.join(dm.sliced_data_dir, 'data')
        print(sliced_dir)
        sample_matrix = load_npy_files_to_matrix(sliced_dir)
        print(sample_matrix.shape)



        cov_matrix = calculate_covariance_matrices(sample_matrix)
        print(cov_matrix.shape)
        mean_cov = get_centroid(cov_matrix)
        print(mean_cov.shape)
        source_centroid_path = os.path.join(os.path.join(sliced_dir, '..'), 'centroid.npy')
        np.save(source_centroid_path, mean_cov)
        plot_covariance_matrix(mean_cov, file_path = f'/home/CoVar_CLISA/cov_ave.png')

if __name__ == "__main__":
    compute_cen()