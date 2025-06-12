import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["WORLD_SIZE"]="1"
import numpy as np
from src.data.io_utils import load_finetune_EEG_data, get_load_data_func, load_processed_SEEDV_NEW_data
from src.data.data_process import running_norm_onesubsession, LDS, LDS_acc, LDS_gpu
from src.data.dataset import ext_Dataset 
import torch
from torch.utils.data import DataLoader
from src.model.MultiModel_PL import MultiModel_PL
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from tqdm import tqdm
import mne
from src.utils import video_order_load, reorder_vids_sepVideo, reorder_vids_back

def normTrain(data,data_train):
    temp = np.transpose(data_train,(0,1,3,2))
    temp = temp.reshape(-1,temp.shape[-1])
    data_mean = np.mean(temp, axis=0)
    data_var = np.var(temp, axis=0)
    data_normed = (data - data_mean.reshape(-1,1)) / np.sqrt(data_var + 1e-5).reshape(-1,1)
    return data_normed

def cal_fea(data,mode):
    if mode == 'de':
        fea = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(data, 3)) + 1.0).squeeze()
        fea[fea<-40] = -40
    elif mode == 'me':
        fea = np.mean(data, axis=3).squeeze()
    return fea

@hydra.main(config_path="cfgs_multi", config_name="config_multi", version_base="1.3")
def ext_fea(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.train.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    load_dir = os.path.join(cfg.data_val.data_dir,'processed_data')
    print('data loading...')
    data, onesub_label, n_samples_onesub, n_samples_sessions = load_finetune_EEG_data(load_dir, cfg.data_val)
    print('data loaded')
    print(f'data ori shape:{data.shape}')
    bs, n_c, n_t = data.shape
    covariances = []
    data = channel_project_numpy(cfg, data, cfg.data_val.channels)
    for i in range(bs):
        # 对每个样本的通道数据计算协方差矩阵
        x = data[i]  # shape: (n_channel, n_times)
        x = x - x.mean(axis=1, keepdims=True)  # 去均值
        cov = np.dot(x, x.T) / (n_t - 1)  # 手动计算协方差
        covariances.append(cov)
    avg_cov = np.mean(covariances, axis=0)
    print(avg_cov.shape)
    save_path = f'./visualize/cov_{cfg.data_val.dataset_name}'
    np.save(save_path,avg_cov)
    print(f'fea saved to {save_path}')

import numpy as np

import numpy as np

def channel_project_numpy(cfg, data, cha_source):
    """
    纯 NumPy 实现：将 EEG 数据从原始通道映射到统一标准通道
    
    参数:
        cfg: 配置对象，需包含 cfg.model.MLLA.uni_channels（标准通道名）
        data: numpy 数组，形状为 (batch_size, n_channel_source, n_timepoint)
        cha_source: list，原始通道名称列表
        
    返回:
        映射后的数据，形状为 (batch_size, n_channel_standard, n_timepoint)
    """
    uni_channelname = cfg.model.MLLA.uni_channels
    channel_interpolate = np.load('channel_interpolate.npy').astype(int)

    batch_size, n_channel_source, n_timepoint = data.shape
    n_channel_standard = len(uni_channelname)

    # 创建源通道名到索引的映射（统一为大写）
    source_ch_map = {name.upper(): idx for idx, name in enumerate(cha_source)}

    # 初始化结果数据
    result = np.zeros((batch_size, n_channel_standard, n_timepoint), dtype=data.dtype)

    # 遍历每个标准通道
    for std_idx, std_name in enumerate(uni_channelname):
        std_name_upper = std_name.upper()

        # Case 1: 通道名直接存在
        if std_name_upper in source_ch_map:
            src_idx = source_ch_map[std_name_upper]
            result[:, std_idx, :] = data[:, src_idx, :]
            continue

        # Case 2: 尝试邻近通道插值
        neighbor_std_indices = channel_interpolate[std_idx]
        valid_src_indices = []

        for neighbor_std_idx in neighbor_std_indices:
            neighbor_std_name = uni_channelname[int(neighbor_std_idx)].upper()
            if neighbor_std_name in source_ch_map:
                valid_src_indices.append(source_ch_map[neighbor_std_name])
                if len(valid_src_indices) == 3:
                    break

        if valid_src_indices:
            neighbor_data = data[:, valid_src_indices, :]  # shape: (batch, M, time)
            interpolated = neighbor_data.mean(axis=1)      # shape: (batch, time)
            result[:, std_idx, :] = interpolated
        else:
            print(f"Channel {std_name} has no available neighbors, filled with zeros")

    return result




if __name__ == '__main__':
    ext_fea()