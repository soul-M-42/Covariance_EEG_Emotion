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
from utils_new import save_batch_images, save_img
from torch.utils.data import DataLoader, TensorDataset, random_split

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
    
    for fold in range(cfg.data_val.n_subs):
        log.info(f"sub:{fold}")
        cp_dir = os.path.join(cfg.log.cp_dir, cfg.data_val.dataset_name, f'r{cfg.log.run}')
        os.makedirs(cp_dir, exist_ok=True)
        wandb_logger = WandbLogger(name=cfg.log.exp_name+'mlp'+'v'+str(cfg.train.valid_method)
                                   +f'_{cfg.data_val.timeLen}_{cfg.data_val.timeStep}_r{cfg.log.run}'+f'_f{fold}', 
                                   project=cfg.log.proj_name, log_model="all")
        cp_monitor = None if n_folds == 1 else "mlp/val/acc"
        es_monitor = "mlp/train/acc" if n_folds == 1 else "mlp/val/acc"
        checkpoint_callback = ModelCheckpoint(monitor=cp_monitor, verbose=True, mode="max", 
                                              dirpath=cp_dir, filename=f'mlp_f{fold}_wd={cfg.mlp.wd}_'+'{epoch}')
        earlyStopping_callback = EarlyStopping(monitor=es_monitor, mode="max", patience=cfg.mlp.patience)
        
        save_dir = os.path.join(cfg.data_val.data_dir,'ext_fea')
        # save_path = os.path.join(save_dir,cfg.log.exp_name+f'_f{fold}_cov_'+cfg.ext_fea.mode+'.npy')
        # cov_fea = np.load(save_path)
        # log.info('cov_fea load from: '+save_path)
        save_path = os.path.join(save_dir,cfg.log.exp_name+f'_sub{fold}_fea_{'pretrain_' if cfg.ext_fea.use_pretrain else ''}{cfg.ext_fea.mode if cfg.ext_fea.use_pretrain else 'DE'}.npy')
        data_sub = np.load(save_path)
        print(data_sub.shape)
        # continue
        log.info('data2 load from: '+save_path)
        # print(data2[:,160])
        if np.isnan(data_sub).any():
            log.warning('nan in data2')
            data_sub = np.where(np.isnan(data_sub), 0, data_sub)
        fea_dim = data_sub.shape[-1]
        data_sub = data_sub.reshape(-1, fea_dim)
        print(f'data_fea.shape:{data_sub.shape}')
        onesub_label2 = np.load(save_dir+'/onesub_label2.npy')
        print(onesub_label2.shape)

        data_tensor = torch.tensor(data_sub, dtype=torch.float32)
        label_tensor = torch.tensor(onesub_label2, dtype=torch.long)

        # Combine data and labels into a TensorDataset
        dataset = TensorDataset(data_tensor, label_tensor)

        # Define split ratios
        train_ratio = 0.8
        test_ratio = 0.2
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size

        # Split the dataset into train and test sets
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Initialize DataLoaders for train and test sets
        trainLoader = DataLoader(train_dataset, batch_size=cfg.mlp.batch_size, shuffle=True)
        valLoader = DataLoader(test_dataset, batch_size=cfg.mlp.batch_size, shuffle=False)

        # continue
        # labels2_train = np.tile(onesub_label2, len(train_subs))
        # labels2_val = np.tile(onesub_label2, len(val_subs))
        # trainset2 = PDataset(data2[train_subs].reshape(-1,data2.shape[-1]), labels2_train)
        # # trainset2 = PDataset(data2[val_subs].reshape(-1,data2.shape[-1]), labels2_val)
        # valset2 = PDataset(data2[val_subs].reshape(-1,data2.shape[-1]), labels2_val)
        # trainLoader = DataLoader(trainset2, batch_size=cfg.mlp.batch_size, shuffle=True, num_workers=cfg.mlp.num_workers)
        # valLoader = DataLoader(valset2, batch_size=cfg.mlp.batch_size, shuffle=False, num_workers=cfg.mlp.num_workers)
        model_mlp = simpleNN3(fea_dim, cfg.mlp.hidden_dim, cfg.mlp.out_dim,0.1)
        predictor = MLPModel(model_mlp, cfg.mlp)
        limit_val_batches = 0.0 if n_folds == 1 else 1.0
        trainer = pl.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback, earlyStopping_callback],
                             max_epochs=cfg.mlp.max_epochs, min_epochs=cfg.mlp.min_epochs,
                             accelerator='gpu', devices=cfg.mlp.gpus, limit_val_batches=limit_val_batches)
        trainer.fit(predictor, trainLoader, valLoader)
        if cfg.train.valid_method != 1:
            best_val_acc_list.append(checkpoint_callback.best_model_score.item())
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