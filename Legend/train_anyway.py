import hydra
from omegaconf import DictConfig
import torch
from model import ExtractorModel, AlignedExtractorModel
import numpy as np
import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
# import wandb
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dual_dataloader import DualDataModule
from dual_model import DualModel_PL
from data.pl_datamodule import EEGDataModule
import logging
from pytorch_lightning.loggers import TensorBoardLogger
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
# os.environ["WORLD_SIZE"]="1"

log = logging.getLogger(__name__)

@hydra.main(config_path="cfgs_dual", config_name="config_dual", version_base="1.3")
def run_pipeline(cfg: DictConfig):
# 1. Load two datasets, with same time_window length, and sample from two datasets
    pl.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    n_folds = cfg.train.n_fold
    for fold in range(n_folds):
        print(f'fold {fold}\n')
        run_name = f'{cfg.log.proj_name}'
        save_dir = os.path.join(os.getcwd(), cfg.log.logpath, run_name, str(fold))
        logger = None
        if not os.path.exists(save_dir):
            if(cfg.log.is_logger):
                os.makedirs(save_dir)
                logger = TensorBoardLogger(save_dir=save_dir, name=run_name)
        # dm = MultiEEGDataModule(cfg.data_1, cfg.data_2, fold, n_folds, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers,
        #                         device=cfg.align.device)
        dm = DualDataModule(cfg.data_1, cfg.data_2, fold, n_folds, num_workers=cfg.train.num_workers,
                            n_pairs=cfg.train.n_pairs,
                            # device=cfg.align.device,
        )
        dm.setup("fit")

    # 2. define channel_specific encoder
        

    # 3. define non-linear covariance alignment module

    # 4. define loss
        model = DualModel_PL(cfg, dm=dm)

        total_params = sum(p.numel() for p in model.parameters())
        total_size = sum(p.numel() * p.element_size() for p in model.parameters())
        log.info(f'Total number of parameters: {total_params}')
        log.info(f'Model size: {total_size} bytes ({total_size / (1024 ** 2):.2f} MB)')
        cp_monitor = None if n_folds == 1 else "ext/val/acc"
        es_monitor = "ext/train/acc" if n_folds == 1 else "ext/val/acc"
        cp_dir = os.path.join(cfg.log.cp_dir, cfg.log.proj_name, f'r{cfg.log.run}')
        checkpoint_callback = ModelCheckpoint(monitor=cp_monitor, mode="max", verbose=True, dirpath=cp_dir, 
                                            filename=f'f{fold}_'+'{epoch}')
        earlyStopping_callback = EarlyStopping(monitor=es_monitor, mode="max", patience=cfg.train.patience)
        limit_val_batches = 0.0 if n_folds == 1 else 1.0
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=cfg.train.max_epochs, min_epochs=cfg.train.min_epochs, 
            accelerator='gpu', devices=cfg.train.gpus, limit_val_batches=limit_val_batches,
            accumulate_grad_batches=cfg.train.grad_accumulation
            )
        trainer.fit(model, dm)

if __name__ == '__main__':
    run_pipeline()