import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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

def normTrain(data: torch.Tensor, data_train: torch.Tensor) -> torch.Tensor:
    """GPU归一化 (全部使用float32)"""
    B, S, C, T = data_train.shape
    
    # 使用float32累加器
    sum_ = torch.zeros(C, dtype=torch.float32, device=data.device)
    sum_sq = torch.zeros(C, dtype=torch.float32, device=data.device)
    
    for t in range(T):
        chunk = data_train[:, :, :, t]  # 已经是float32
        sum_ += chunk.sum(dim=(0,1))
        sum_sq += (chunk ** 2).sum(dim=(0,1))
    
    total = B * S * T
    data_mean = sum_ / total
    data_var = (sum_sq / total) - (data_mean ** 2)
    
    # 分块归一化
    for c in range(C):
        data[:, :, c, :] = (data[:, :, c, :] - data_mean[c]) / torch.sqrt(data_var[c] + 1e-5)
    
    return data

def cal_fea(data: torch.Tensor, mode: str) -> torch.Tensor:
    """GPU-based feature calculation (float32)"""
    if mode == 'de':
        fea = 0.5 * torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0, device=data.device)) * 
                             (torch.var(data, dim=3)) + 1.0).squeeze()
        fea = torch.clamp(fea, min=-40.0)
    elif mode == 'me':
        fea = torch.mean(data, dim=3).squeeze()
    return fea

@hydra.main(config_path="cfgs_multi", config_name="config_multi", version_base="1.3")
def ext_fea(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.train.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据并转换为float32 GPU张量
    load_dir = os.path.join(cfg.data_val.data_dir,'processed_data')
    print('Loading data...')
    data_np, onesub_label, n_samples_onesub, n_samples_sessions = load_finetune_EEG_data(load_dir, cfg.data_val)
    data = torch.from_numpy(data_np).float().to(device)  # 默认float32
    data = data.view(cfg.data_val.n_subs, -1, data.shape[-2], data.shape[-1])
    del data_np
    
    # 转换其他数据到GPU (保持整数类型)
    n_samples_onesub = torch.from_numpy(n_samples_onesub).to(device)
    n_samples_sessions = torch.from_numpy(n_samples_sessions).to(device)
    
    save_dir = os.path.join(cfg.data_val.data_dir,'ext_fea')
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'onesub_label.npy'), onesub_label)
    
    # 初始化模型
    if cfg.val.extractor.use_pretrain:
        cp_path = os.path.join('log', cfg.log.run_name, 'ckpt', f'epoch={(cfg.val.extractor.ckpt_epoch-1):02d}.ckpt')
        cfg.data_cfg_list = [cfg.data_0, cfg.data_1, cfg.data_2, cfg.data_3, cfg.data_4, cfg.data_val]
        cfg.data_cfg_list = [cfg_i for cfg_i in cfg.data_cfg_list if cfg_i.dataset_name != 'None']
        Extractor = MultiModel_PL.load_from_checkpoint(checkpoint_path=cp_path, cfg=cfg, strict=False).to(device)
        Extractor.save_fea = True
        Extractor.cnn_encoder.set_saveFea(True)
        trainer = pl.Trainer(accelerator='gpu', devices=1)
    
    # 主处理循环
    n_folds = len(cfg.data_val.val_subs_all) if cfg.val.extractor.normTrain else 1
    for fold in tqdm(range(n_folds), desc='Extracting features'):
        if cfg.val.extractor.normTrain:
            val_subs = cfg.data_val.val_subs_all[fold]
            train_subs = list(set(range(cfg.data_val.n_subs)) - set(val_subs))
        else:
            train_subs, val_subs = [], []
        
        data_train = data[train_subs] if train_subs else data
        data_fold = normTrain(data, data_train) if cfg.val.extractor.normTrain else data
        
        if cfg.val.extractor.use_pretrain:
            data_fold = data_fold.view(-1, data_fold.shape[-2], data_fold.shape[-1])
            dataset = ext_Dataset(data_fold.cpu().numpy(), np.tile(onesub_label, cfg.data_val.n_subs))
            fold_loader = DataLoader(dataset, batch_size=cfg.val.extractor.batch_size, shuffle=False, 
                                   num_workers=cfg.train.num_workers, pin_memory=True)
            with torch.no_grad():
                pred = torch.cat(trainer.predict(Extractor, fold_loader), dim=0).to(device)
            fea = cal_fea(pred, cfg.val.extractor.fea_mode)
        else:
            fea = compute_de_features_gpu(data_fold, cfg, device)  
        
        fea_train = fea[train_subs]
        data_mean = torch.mean(fea_train.view(-1, fea_train.shape[-1]), dim=0)
        data_var = torch.var(fea_train.view(-1, fea_train.shape[-1]), dim=0)
        
        if cfg.val.extractor.normTrain:
            n_sample_sum_sessions_cum = torch.cat([torch.tensor([0], device=device), 
                                                 torch.cumsum(n_samples_sessions.sum(1), dim=0)])
            for sub in range(cfg.data_val.n_subs):
                for s in range(len(n_samples_sessions)):
                    slice_idx = slice(n_sample_sum_sessions_cum[s], n_sample_sum_sessions_cum[s+1])
                    fea[sub, slice_idx] = running_norm_onesubsession_gpu(
                        fea[sub, slice_idx], data_mean, data_var, cfg.val.extractor.rn_decay)
        
        if cfg.val.extractor.LDS:
            n_samples_onesub_cum = torch.cat([torch.tensor([0], device=device), 
                                            torch.cumsum(n_samples_onesub, dim=0)])
            for sub in range(cfg.data_val.n_subs):
                for vid in range(len(n_samples_onesub)):
                    slice_idx = slice(n_samples_onesub_cum[vid], n_samples_onesub_cum[vid+1])
                    fea[sub, slice_idx] = LDS_gpu(fea[sub, slice_idx])
        
        fea_np = fea.reshape(-1, fea.shape[-1]).cpu().numpy()
        save_path = os.path.join(save_dir, f"{cfg.log.run_name}_f{fold}_fea.npy")
        np.save(save_path, fea_np)
        print(f"Features saved to {save_path}")

def compute_de_features_gpu(data: torch.Tensor, cfg, device) -> torch.Tensor:
    """GPU-based differential entropy (float32)"""
    # 实现具体的DE计算逻辑
    pass

def running_norm_onesubsession_gpu(data: torch.Tensor, 
                                 data_mean: torch.Tensor,
                                 data_var: torch.Tensor,
                                 decay_rate: float) -> torch.Tensor:
    """GPU滑动归一化 (float32)"""
    data_norm = torch.zeros_like(data)
    running_sum = torch.zeros(data.size(1), dtype=torch.float32, device=data.device)
    running_square = torch.zeros_like(running_sum)
    decay_factor = 1.0
    
    for counter in range(data.size(0)):
        data_one = data[counter]
        running_sum += data_one
        running_mean = running_sum / (counter + 1)
        running_square += data_one ** 2
        running_var = (running_square / (counter + 1)) - running_mean ** 2
        
        curr_mean = decay_factor * data_mean + (1 - decay_factor) * running_mean
        curr_var = decay_factor * data_var + (1 - decay_factor) * torch.clamp(running_var, min=0)
        decay_factor *= decay_rate
        
        data_norm[counter] = (data_one - curr_mean) / torch.sqrt(curr_var + 1e-5)
        data_norm[counter] = torch.clamp(data_norm[counter], min=-5.0, max=5.0)
    
    return data_norm

if __name__ == '__main__':
    ext_fea()