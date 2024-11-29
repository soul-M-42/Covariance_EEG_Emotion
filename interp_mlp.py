import hydra
from omegaconf import DictConfig
from model.models import simpleNN3
import numpy as np
import os
from data.dataset import PDataset
from model.pl_models import MLPModel
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import torch
import logging
from captum.attr import IntegratedGradients

log = logging.getLogger(__name__)

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def interp_mlp(cfg: DictConfig) -> None:
    
    pl.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if isinstance(cfg.train.valid_method, int):
        n_folds = cfg.train.valid_method
    elif cfg.train.valid_method == 'loo':
        n_folds = cfg.train.n_subs

    n_per = round(cfg.data.n_subs / n_folds)
    best_val_acc_list = []
    
    target_label = 0
    device = torch.device('cuda')
    for fold in range(0,1):
        cp_dir = os.path.join(cfg.log.cp_dir, cfg.data.dataset_name)
        wandb_logger = WandbLogger(name=cfg.log.exp_name+'mlp'+'v'+str(cfg.train.valid_method)
                                   +f'_{cfg.data.timeLen}_{cfg.data.timeStep}_r{cfg.log.run}'+f'_f{fold}', 
                                   project=cfg.log.proj_name, log_model="all")
        checkpoint_callback = ModelCheckpoint(monitor="mlp/val/acc", mode="max", dirpath=cp_dir, filename=cfg.log.exp_name+'_mlp_r'+str(cfg.log.run)+f'_f{fold}_best')
        earlyStopping_callback = EarlyStopping(monitor="mlp/val/acc", mode="max", patience=cfg.mlp.patience)
        log.info(f"fold:{fold}")
        if n_folds == 1:
            val_subs = []
        elif fold < n_folds - 1:
            val_subs = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_subs = np.arange(n_per * fold, cfg.data.n_subs)            
        train_subs = list(set(np.arange(cfg.data.n_subs)) - set(val_subs))
        # if len(val_subs) == 1:
        #     val_subs = list(val_subs) + train_subs
        log.info(f'train_subs:{train_subs}')
        log.info(f'val_subs:{val_subs}')
        
        save_dir = os.path.join(cfg.data.data_dir,'ext_fea',f'fea_r{cfg.log.run}')
        save_path = os.path.join(save_dir,cfg.log.exp_name+'_r'+str(cfg.log.run)+f'_f{fold}_fea_'+cfg.ext_fea.mode+'.npy')
        data2 = np.load(save_path)
        # print(data2[:,160])
        if np.isnan(data2).any():
            log.warning('nan in data2')
            data2 = np.where(np.isnan(data2), 0, data2)
        data2 = data2.reshape(cfg.data.n_subs, -1, data2.shape[-1])
        onesub_label2 = np.load(save_dir+'/onesub_label2.npy')
        labels2_train = np.tile(onesub_label2, len(train_subs))
        labels2_val = np.tile(onesub_label2, len(val_subs))
        trainset2 = PDataset(data2[train_subs].reshape(-1,data2.shape[-1]), labels2_train)
        # trainset2 = PDataset(data2[val_subs].reshape(-1,data2.shape[-1]), labels2_val)
        valset2 = PDataset(data2[val_subs].reshape(-1,data2.shape[-1]), labels2_val)
        trainLoader = DataLoader(trainset2, batch_size=cfg.mlp.batch_size, shuffle=True, num_workers=cfg.mlp.num_workers)
        valLoader = DataLoader(valset2, batch_size=cfg.mlp.batch_size, shuffle=False, num_workers=cfg.mlp.num_workers)
        model_mlp = simpleNN3(cfg.mlp.fea_dim, cfg.mlp.hidden_dim, cfg.mlp.out_dim,0.1)
        
        # load checkpoint
        checkpoint = torch.load(os.path.join(cfg.data.param_dir, 'segatt_15_mlp_r13_f0_best.ckpt'))
        model_weights = checkpoint['state_dict']
        model_mlp.load_state_dict(model_weights)
        model_mlp.eval()
        
        n_samples_all = len(valset2)
        attributions_all = torch.zeros((n_samples_all * len(val_subs), 256))
        for counter, (x_batch, y_batch) in enumerate(valLoader):
            print('counter:', counter)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            baseline = torch.zeros_like(x_batch).to(device)

            ig = IntegratedGradients(model_mlp)
            attributions, delta = ig.attribute(x_batch, baseline, target=target_label, return_convergence_delta=True)
            attributions_all[n_samples_all * counter: n_samples_all * (counter+1), :] = attributions.cpu()
        
        attrs = attributions_all.numpy()

        if not os.path.exists(os.path.join(cfg.data.param_dir, 'importance')):
            os.makedirs(os.path.join(cfg.data.param_dir, 'importance'))
        np.save(os.path.join(cfg.data.param_dir, 'importance', 'attrs_fold0_cls0.npy'), attrs)
        wandb.finish()
        
if __name__ == '__main__':
    interp_mlp()