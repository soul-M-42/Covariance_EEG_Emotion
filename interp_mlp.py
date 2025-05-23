import hydra
from omegaconf import DictConfig
from src.model.valMLP import simpleNN3
import numpy as np
import os
from src.data.dataset import PDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import torch
import logging
from captum.attr import IntegratedGradients

log = logging.getLogger(__name__)

def map_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "")
        new_state_dict[new_key] = value
    return new_state_dict

@hydra.main(config_path="cfgs_multi", config_name="config_multi", version_base="1.3")
def interp_mlp(cfg: DictConfig) -> None:
    
    pl.seed_everything(cfg.val.mlp.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # if isinstance(cfg.train.valid_method, int):
    #     n_folds = cfg.train.valid_method
    # elif cfg.train.valid_method == 'loo':
    #     n_folds = cfg.train.n_subs

    n_folds = 1
    n_per = round(cfg.data_val.n_subs / n_folds)    
    
    device = torch.device('cuda')
    
    attributions_all = []
    for target_label in range(cfg.data_val.n_class):
        print('\n\nTarget label:', target_label)
        fold = 0
        cp_dir = os.path.join(cfg.log.mlp_cp_dir, cfg.log.run_name)
        # checkpoint_callback = ModelCheckpoint(monitor="mlp/val/acc", mode="max", dirpath=cp_dir, filename=cfg.log.exp_name+'_mlp_r'+str(cfg.log.run)+f'_f{fold}_best')
        # earlyStopping_callback = EarlyStopping(monitor="mlp/val/acc", mode="max", patience=cfg.val.mlp.patience)
        checkpoint_callback = ModelCheckpoint(
            monitor="mlp/val/acc", verbose=True, mode="max",
            dirpath=cp_dir,
            filename=f'mlp_f{fold}_wd={cfg.val.mlp.wd}_{{epoch}}',
            save_top_k=1,
        )
        log.info(f"fold:{fold}")
        # if n_folds == 1:
        #     val_subs = []
        # elif fold < n_folds - 1:
        #     val_subs = np.arange(n_per * fold, n_per * (fold + 1))
        # else:
        #     val_subs = np.arange(n_per * fold, cfg.data_val.n_subs)            
        # train_subs = list(set(np.arange(cfg.data_val.n_subs)) - set(val_subs))
        val_subs = np.arange(cfg.data_val.n_subs)
        train_subs = val_subs
        # if len(val_subs) == 1:
        #     val_subs = list(val_subs) + train_subs
        log.info(f'train_subs:{train_subs}')
        log.info(f'val_subs:{val_subs}')
        
        save_dir = os.path.join(cfg.data_val.data_dir, 'ext_fea')
        if not cfg.val.extractor.normTrain:
            save_path = os.path.join(save_dir,cfg.log.run_name+f'_all_fea_{f'epoch={(cfg.val.extractor.ckpt_epoch-1):02d}.ckpt' if cfg.val.extractor.use_pretrain else ""}{cfg.val.extractor.fea_mode if cfg.val.extractor.use_pretrain else cfg.val.extractor.fea_mode}.npy')
        else:
            save_path = os.path.join(save_dir,cfg.log.run_name+f'_f{fold}_fea_{f'epoch={(cfg.val.extractor.ckpt_epoch-1):02d}.ckpt' if cfg.val.extractor.use_pretrain else ""}{cfg.val.extractor.fea_mode if cfg.val.extractor.use_pretrain else cfg.val.extractor.fea_mode}.npy')
        print(f'loading from {save_path}')
        data2 = np.load(save_path)
        # print(data2[:,160])
        if np.isnan(data2).any():
            log.warning('nan in data2')
            data2 = np.where(np.isnan(data2), 0, data2)
        data2 = data2.reshape(cfg.data_val.n_subs, -1, data2.shape[-1])
        onesub_label2 = np.load(save_dir+'/onesub_label.npy')
        labels2_train = np.tile(onesub_label2, len(train_subs))
        labels2_val = np.tile(onesub_label2, len(val_subs))
        trainset2 = PDataset(data2[train_subs].reshape(-1,data2.shape[-1]), labels2_train)
        # trainset2 = PDataset(data2[val_subs].reshape(-1,data2.shape[-1]), labels2_val)
        valset2 = PDataset(data2[val_subs].reshape(-1,data2.shape[-1]), labels2_val)
        trainLoader = DataLoader(trainset2, batch_size=cfg.val.mlp.batch_size, shuffle=True)
        valLoader = DataLoader(valset2, batch_size=len(onesub_label2), shuffle=False)
        fea_dim = data2.shape[-1]
        model_mlp = simpleNN3(fea_dim, cfg.val.mlp.hidden_dim, cfg.val.mlp.out_dim,0.1).to(device)
        
        cp_epoch = 7
        # load checkpoint
        checkpoint = torch.load(os.path.join(cp_dir, f'{cfg.data_val.dataset_name}_mlp_f{fold if cfg.val.n_fold != 'inter' else '_inter'}_wd={cfg.val.mlp.wd}_epoch=2.ckpt'), map_location='cuda:0')
        model_weights = checkpoint['state_dict']
        model_weights_mapped = map_keys(model_weights)

        model_mlp.load_state_dict(model_weights_mapped)
        model_mlp.eval()
        
        n_samples_all = len(onesub_label2)
        print(n_samples_all)
        att_emo = []
        for counter, (x_batch, y_batch) in enumerate(valLoader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            baseline = torch.zeros_like(x_batch).to(device)

            ig = IntegratedGradients(model_mlp)
            attributions, delta = ig.attribute(x_batch, baseline, target=target_label, return_convergence_delta=True)
            # attributions = [n_fea_per_sub, dim_fea]
            att_emo.append(attributions.cpu())
        attributions_all.append(att_emo)
            
    attrs = np.array(attributions_all)
    print(attrs.shape)

    att_save_dir = os.path.join('visualize', 'importance', cfg.log.run_name, cfg.data_val.dataset_name)
    if not os.path.exists(att_save_dir):
        os.makedirs(att_save_dir)

    np.save(os.path.join(att_save_dir, 'attrs_allData_cls_all.npy'), attrs)
            
        
if __name__ == '__main__':
    interp_mlp()