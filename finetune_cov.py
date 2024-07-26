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
log = logging.getLogger(__name__)


@hydra.main(config_path="cfgs", config_name="config_cov", version_base="1.3")
def finetune_dataset(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if isinstance(cfg.train.valid_method, int):
        n_folds = cfg.train.valid_method
    elif cfg.train.valid_method == 'loo':
        n_folds = cfg.train.n_subs
    n_per = round(cfg.data.n_subs / n_folds)

    for fold in range(0,n_folds):
        # load SEED-V data

        print("fold:", fold)
        cp_dir = os.path.join(cfg.log.cp_dir, cfg.data.dataset_name, f'r{cfg.log.run}')
        os.makedirs(cp_dir, exist_ok=True)
        wandb_logger = WandbLogger(name=cfg.log.exp_name+'v'+str(cfg.train.valid_method)
                                   +f'_{cfg.data.timeLen}_{cfg.data.timeStep}_r{cfg.log.run}'+f'_f{fold}', 
                                   project=cfg.log.proj_name, log_model="all")
        checkpoint_callback = ModelCheckpoint(monitor="ext/val/acc", mode="max", dirpath=cp_dir, filename=f'f{fold}'+'{epoch}')
        earlyStopping_callback = EarlyStopping(monitor="ext/val/acc", mode="max", patience=cfg.train.patience)
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

        # few_subject transfer learning
        tmp_subs = train_subs
        train_subs = val_subs
        val_subs = tmp_subs
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
        # if cfg.data.dataset_name == 'SEEDV':
        #     dm = SEEDVDataModule(load_dir, save_dir, cfg.data.timeLen, cfg.data.timeStep, train_subs,
        #                          val_subs, train_vids, val_vids, cfg.data.n_session, cfg.data.fs, 
        #                          cfg.data.n_channs, cfg.data.n_subs, cfg.data.n_vids, cfg.data.n_class,
        #                          cfg.train.valid_method=='loo', cfg.train.num_workers)
        # elif cfg.data.dataset_name == 'FACED':
        #     dm = FACEDDataModule(load_dir, save_dir, cfg.data.timeLen, cfg.data.timeStep, train_subs,
        #                          val_subs, train_vids, val_vids, cfg.data.n_session, cfg.data.fs, 
        #                          cfg.data.n_channs, cfg.data.n_subs, cfg.data.n_vids, cfg.data.n_class,
        #                          cfg.train.valid_method=='loo', cfg.train.num_workers)
        # elif cfg.data.dataset_name == 'SEED':
        #     pass
        dm = EEGDataModule(cfg.data, train_subs, val_subs, train_vids, val_vids,
                           cfg.train.valid_method=='loo', cfg.train.num_workers)
            

        
        # load model
        # model = hydra.utils.instantiate(cfg.model)

        # total_params = sum(p.numel() for p in model.parameters())
        # total_size = sum(p.numel() * p.element_size() for p in model.parameters())
        # log.info(f'Total number of parameters: {total_params}')
        # log.info(f'Model size: {total_size} bytes ({total_size / (1024 ** 2):.2f} MB)')
        
        # Extractor = ExtractorModel(model, cfg.train)

        checkpoint =  os.path.join(cfg.log.cp_dir,cfg.data_source.dataset_name,f'r{cfg.log.run}',f'f{fold}*')
        checkpoint = glob.glob(checkpoint)[0]
        
        log.info('checkpoint load from: '+checkpoint)
        Extractor = ExtractorModel.load_from_checkpoint(checkpoint_path=checkpoint)
        Extractor = AlignedExtractorModel(Extractor, dm, cfg)
        Extractor.set_freeze_ext(freeze=True)
        # Extractor.set_freeze_align(freeze=True)
        
        print(Extractor)
        Extractor.model.stratified = []

        limit_val_batches = 0.0 if n_folds == 1 else 1.0
        trainer = pl.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback, earlyStopping_callback],
                             max_epochs=cfg.train.max_epochs, min_epochs=cfg.train.min_epochs, 
                             accelerator='gpu', devices=cfg.train.gpus, limit_val_batches=limit_val_batches)
        trainer.fit(Extractor, dm)
        wandb.finish()
        
        if cfg.train.iftest :
            break

    # train


    # set logger
   
    # set seed

if __name__ == "__main__":
    finetune_dataset()
