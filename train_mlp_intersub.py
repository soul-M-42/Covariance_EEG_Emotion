import hydra
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ["WORLD_SIZE"]="1"
from omegaconf import DictConfig
from model.models import simpleNN3
import numpy as np
from data.dataset import PDataset
from model.pl_models import MLPModel
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import torch
import logging
from utils_new import save_batch_images, save_img


log = logging.getLogger(__name__)

@hydra.main(config_path="cfgs_multi", config_name="config_multi", version_base="1.3")
def train_mlp(cfg: DictConfig) -> None:
    
    pl.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if isinstance(cfg.train.valid_method, int):
        n_folds = cfg.finetune.valid_method
    elif cfg.train.valid_method == 'loo':
        n_folds = cfg.data_val.n_subs

    n_per = round(cfg.data_val.n_subs / n_folds)
    best_val_acc_list = []
    
    for fold in range(0,n_folds):
        fold_acc_max = 0
        cp_dir = os.path.join(cfg.log.cp_dir, cfg.data_val.dataset_name, f'r{cfg.log.run}')
        os.makedirs(cp_dir, exist_ok=True)
        wandb_logger = WandbLogger(name=cfg.log.exp_name+'mlp'+'v'+str(cfg.train.valid_method)
                                   +f'_{cfg.data_val.timeLen}_{cfg.data_val.timeStep}_r{cfg.log.run}'+f'_f{fold}', 
                                   project=cfg.log.proj_name, log_model="all")
        cp_monitor = None if n_folds == 1 else "mlp/val/acc"
        es_monitor = "mlp/train/acc" if n_folds == 1 else "mlp/val/acc"

        earlyStopping_callback = EarlyStopping(monitor=es_monitor, mode="max", patience=cfg.mlp.patience)
        log.info(f"fold:{fold}")
        if n_folds == 1:
            val_subs = []
        elif fold < n_folds - 1:
            val_subs = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_subs = np.arange(n_per * fold, cfg.data_val.n_subs)            
        train_subs = list(set(np.arange(cfg.data_val.n_subs)) - set(val_subs))
        # if len(val_subs) == 1:
        #     val_subs = list(val_subs) + train_subs
        # train_subs = finetune_subs
        if cfg.train.reverse:
            train_subs, val_subs = val_subs, train_subs
        log.info(f'finetune_subs:{train_subs}')
        log.info(f'val_subs:{val_subs}')
        
        save_dir = os.path.join(cfg.data_val.data_dir,'ext_fea')
        # save_path = os.path.join(save_dir,cfg.log.exp_name+f'_f{fold}_cov_'+cfg.ext_fea.mode+'.npy')
        # cov_fea = np.load(save_path)
        # log.info('cov_fea load from: '+save_path)
        save_path = os.path.join(save_dir,cfg.log.exp_name+f'_f{fold}_fea_{"pretrain_" if cfg.ext_fea.use_pretrain else ""}{cfg.ext_fea.mode if cfg.ext_fea.use_pretrain else "DE"}.npy')
        data = np.load(save_path)
        log.info('data load from: '+save_path)
        # print(data[:,160])
        if np.isnan(data).any():
            log.warning('nan in data')
            data = np.where(np.isnan(data), 0, data)
        fea_dim = data.shape[-1]
        n_sub = cfg.data_val.n_subs
        data = data.reshape(n_sub, -1, data.shape[-1])
        n_fea = data.shape[1]
        print(f'data_fea.shape:{data.shape}')
        # [sub, n_fea, dim_fea]
        # save_batch_images(data[:, :1000, :], 'fea_mlp')
        onesub_label = np.load(save_dir+'/onesub_label2.npy')
        labels_train = np.tile(onesub_label, len(train_subs))
        labels_val = np.tile(onesub_label, len(val_subs))
        sub_val_acc_best = []
        for i_sub in range(n_sub):
            checkpoint_callback = ModelCheckpoint(monitor=cp_monitor, verbose=True, mode="max", 
                                        dirpath=cp_dir, filename=f'mlp_f{fold}_wd={cfg.mlp.wd}_'+'{epoch}',
                                        save_top_k=1,
                                        )
            trainset = PDataset(data[i_sub, :n_fea//2].reshape(-1,fea_dim), onesub_label[:n_fea//2])
            valset = PDataset(data[i_sub, n_fea//2:].reshape(-1,fea_dim), onesub_label[n_fea//2:])
            # testset = PDataset(data[i_sub, (n_fea//3+n_fea//3):].reshape(-1,fea_dim), onesub_label[(n_fea//3+n_fea//3):])
            trainLoader = DataLoader(trainset, batch_size=cfg.mlp.batch_size, shuffle=True, num_workers=cfg.mlp.num_workers)
            valLoader = DataLoader(valset, batch_size=cfg.mlp.batch_size, shuffle=False, num_workers=cfg.mlp.num_workers)
            # testLoader = DataLoader(testset, batch_size=cfg.mlp.batch_size, shuffle=False, num_workers=cfg.mlp.num_workers)

        
            model_mlp = simpleNN3(fea_dim, cfg.mlp.hidden_dim, cfg.mlp.out_dim,0.1)
            predictor = MLPModel(model_mlp, cfg.mlp)
            limit_val_batches = 0.0 if n_folds == 1 else 1.0
            trainer = pl.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback],
                                max_epochs=cfg.mlp.max_epochs, min_epochs=cfg.mlp.min_epochs,
                                accelerator='gpu', devices=cfg.mlp.gpus, limit_val_batches=limit_val_batches)
            trainer.fit(predictor, trainLoader, valLoader)
            print(f'Sub {i_sub} val_best = {trainer.checkpoint_callback.best_model_score.detach().cpu().numpy()}')
            sub_val_acc_best.append(trainer.checkpoint_callback.best_model_score.detach().cpu().numpy())
        print(f'被试内平均acc:{np.mean(sub_val_acc_best)}')
        # if cfg.train.valid_method != 1:
        #     best_val_acc_list.append(trainer.callback_metrics['mlp/val/acc'].detach().cpu().numpy())
        wandb.finish()
        
        if cfg.train.iftest :
            break
    if cfg.train.valid_method != 1:
        log.info("Best train/validation accuracies for each fold:")
        for fold, acc in enumerate(best_val_acc_list):
            log.info(f"    Fold {fold}: {acc}")
        
        average_val_acc = np.mean(best_val_acc_list)
        log.info(f"Average train/validation accuracy across all folds: {average_val_acc}")
        std_val_acc = np.std(best_val_acc_list)
        log.info(f"Standard deviation of train/validation accuracy across all folds: {std_val_acc}")
        log.info(f"Extracting features with {cfg.mlp.wd}: $mlp_wd and ext_wd: {cfg.train.wd}")

if __name__ == '__main__':
    train_mlp()