import hydra
from omegaconf import DictConfig
import torch
from model import ExtractorModel
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.data.multi_dataloader import MultiDataModule
from src.model.MultiModel_PL import MultiModel_PL
from data.pl_datamodule import EEGDataModule
import logging

log = logging.getLogger(__name__)

def update_cfg_with_wandb(cfg, wandb_config):
    """
    Update the cfg DictConfig with values from wandb.config
    """
    for key, value in wandb_config.items():
        if isinstance(value, dict):
            if key in cfg:
                update_cfg_with_wandb(cfg[key], value)
            else:
                cfg[key] = value
        else:
            cfg[key] = value
    return cfg

@hydra.main(config_path="cfgs_multi", config_name="config_multi", version_base="1.3")
def run_pipeline(cfg: DictConfig):
    # 1. Initialize wandb and update cfg
    wandb.init(project=cfg.log.proj_name, config=cfg)
    cfg = update_cfg_with_wandb(cfg, wandb.config)
    
    # Set seed and deterministic flags
    pl.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n_folds = cfg.train.n_fold
    for fold in range(n_folds):
        print(f'fold {fold}\n')
        run_name = f'{cfg.log.proj_name}'
        save_dir = os.path.join(os.getcwd(), cfg.log.logpath, run_name, str(fold))
        
        # 2. Set up WandbLogger
        logger = WandbLogger(project=cfg.log.proj_name, name=run_name, save_dir=save_dir)
        
        n_per = round(cfg.data_val.n_subs / n_folds)
        if n_folds == 1:
            val_subs = []
        elif fold < n_folds - 1:
            val_subs = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_subs = np.arange(n_per * fold, cfg.data_val.n_subs)
        train_subs = list(set(np.arange(cfg.data_val.n_subs)) - set(val_subs))
        if len(val_subs) == 1:
            val_subs = list(val_subs) + train_subs
        print('train_subs:', train_subs)
        print('val_subs:', val_subs)
        
        if cfg.data_val.dataset_name == 'FACED':
            n_vids = 24 if cfg.data_val.n_class == 2 else 28
        else:
            n_vids = cfg.data_val.n_vids
        train_vids = np.arange(n_vids)
        val_vids = np.arange(n_vids)

        dm = EEGDataModule(cfg.data_val, train_subs, val_subs, train_vids, val_vids,
                           cfg.train.valid_method == 'loo', cfg.train.num_workers)
        
        dm.setup("fit")
        
        # Initialize the model
        model = MultiModel_PL(cfg)
        
        total_params = sum(p.numel() for p in model.parameters())
        total_size = sum(p.numel() * p.element_size() for p in model.parameters())
        log.info(f'Total number of parameters: {total_params}')
        log.info(f'Model size: {total_size} bytes ({total_size / (1024 ** 2):.2f} MB)')
        
        cp_monitor = None if n_folds == 1 else "loss_total/val"
        es_monitor = "loss_total/train" if n_folds == 1 else "loss_total/val"
        cp_dir = os.path.join(cfg.log.logpath, cfg.log.proj_name)
        
        checkpoint_callback = ModelCheckpoint(monitor=cp_monitor, mode="min", verbose=True, dirpath=cp_dir, 
                                              filename=f'f{fold}_'+'{epoch}')
        earlyStopping_callback = EarlyStopping(monitor=es_monitor, mode="min", patience=cfg.train.patience)
        limit_val_batches = 0.0 if n_folds == 1 else 1.0
        
        # 3. Set up the trainer with WandbLogger
        trainer = pl.Trainer(
            logger=logger,
            callbacks=[checkpoint_callback, earlyStopping_callback],
            max_epochs=cfg.train.max_epochs, min_epochs=cfg.train.min_epochs,
            accelerator='gpu', devices=cfg.train.gpus, limit_val_batches=limit_val_batches,
            accumulate_grad_batches=cfg.train.grad_accumulation
        )
        
        # Start training
        trainer.fit(model, dm)

if __name__ == '__main__':
    run_pipeline()
