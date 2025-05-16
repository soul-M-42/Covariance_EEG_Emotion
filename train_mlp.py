import hydra
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["WORLD_SIZE"]="1"
from omegaconf import DictConfig
from src.model.valMLP import simpleNN3
import numpy as np
from src.data.dataset import PDataset
from src.model.valMLP import MLPModel
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import torch
import logging
from src.utils import save_batch_images, save_img

@hydra.main(config_path="cfgs_multi", config_name="config_multi", version_base="1.3")
def train_mlp(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.val.mlp.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    val_subs_all = cfg.data_val.val_subs_all
    if cfg.val.n_fold == "loo":
        val_subs_all = [[i] for i in range(cfg.data_val.n_subs)]
    n_folds = len(val_subs_all)
    best_val_acc_list = []
    for fold in range(0,n_folds):
        cp_dir = os.path.join(cfg.log.mlp_cp_dir, cfg.log.run_name)
        os.makedirs(cp_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(monitor="mlp/val/acc",
                                              verbose=True, mode="max", 
                                              dirpath=cp_dir, filename=f'mlp_f{fold}_wd={cfg.val.mlp.wd}_'+'{epoch}',
                                              save_top_k=1,
                                              )
        print(f"fold:{fold}")
        val_subs = val_subs_all[fold]
        train_subs = list(set(np.arange(cfg.data_val.n_subs)) - set(val_subs))
        if cfg.val.extractor.reverse:
            train_subs, val_subs = val_subs, train_subs
        print(f'finetune_subs:{train_subs}')
        print(f'val_subs:{val_subs}')
        save_dir = os.path.join(cfg.data_val.data_dir,'ext_fea')
        save_path = os.path.join(save_dir,cfg.log.run_name+f'_f{fold}_fea_{f'epoch={(cfg.val.extractor.ckpt_epoch-1):02d}.ckpt' if cfg.val.extractor.use_pretrain else ""}{cfg.val.extractor.fea_mode if cfg.val.extractor.use_pretrain else cfg.val.extractor.fea_mode}.npy')
        data = np.load(save_path)
        print('fea data load from: '+ save_path)
        if np.isnan(data).any():
            print('nan in data')
            data = np.where(np.isnan(data), 0, data)
        fea_dim = data.shape[-1]
        data = data.reshape(cfg.data_val.n_subs, -1, data.shape[-1])
        print(f'data_fea.shape:{data.shape}')
        onesub_label = np.load(save_dir+'/onesub_label.npy')
        labels_train = np.tile(onesub_label, len(train_subs))
        labels_val = np.tile(onesub_label, len(val_subs))
        trainset = PDataset(data[train_subs].reshape(-1,data.shape[-1]), labels_train)
        valset = PDataset(data[val_subs].reshape(-1,data.shape[-1]), labels_val)
        trainLoader = DataLoader(trainset, batch_size=cfg.val.mlp.batch_size, shuffle=True)
        valLoader = DataLoader(valset, batch_size=cfg.val.mlp.batch_size, shuffle=False)
        model_mlp = simpleNN3(fea_dim, cfg.val.mlp.hidden_dim, cfg.val.mlp.out_dim,0.1)
        predictor = MLPModel(model_mlp, cfg.val.mlp)
        trainer = pl.Trainer(callbacks=[checkpoint_callback],
                             max_epochs=cfg.val.mlp.max_epochs, min_epochs=cfg.val.mlp.min_epochs,
                             accelerator='gpu', devices=1, limit_val_batches=1.0)
        trainer.fit(predictor, trainLoader, valLoader)
        bese_acc = trainer.checkpoint_callback.best_model_score.detach().cpu().numpy()
        best_val_acc_list.append(bese_acc)
        print(f'Best acc on fold {fold} = {bese_acc}')
    print("Best train/validation accuracies for each fold:")
    for fold, acc in enumerate(best_val_acc_list):
        print(f"Fold {fold}: {acc}")
    average_val_acc = np.mean(best_val_acc_list)
    std_val_acc = np.std(best_val_acc_list)
    print(f'Average: {average_val_acc} ± {std_val_acc}')

if __name__ == '__main__':
    train_mlp()