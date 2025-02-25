 
import numpy as np
from data.io_utils import load_finetune_EEG_data, get_load_data_func, load_processed_SEEDV_NEW_data
from data.data_process import running_norm_onesubsession, LDS, LDS_acc, LDS_gpu
from utils.reorder_vids import video_order_load, reorder_vids_sepVideo, reorder_vids_back
import hydra
from omegaconf import DictConfig
from model import ExtractorModel
from src.model.MultiModel_PL import MultiModel_PL
from data.dataset import SEEDV_Dataset 
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import os
from tqdm import tqdm
import logging
import mne
import glob
from utils_new import save_batch_images, save_img

log = logging.getLogger(__name__)

@hydra.main(config_path="cfgs_multi", config_name="config_multi", version_base="1.3")
def ext_fea(cfg: DictConfig) -> None:
    load_dir = os.path.join(cfg.data_val.data_dir,'processed_data')
    print('data loading...')
    data_all, onesub_label2, n_samples2_onesub, n_samples2_sessions = load_finetune_EEG_data(load_dir, cfg.data_val)
    print('data loaded')
    print(f'data ori shape:{data_all.shape}')
    #data2 shape (n_subs,session*vid*n_samples, n_chans, n_pionts)
    data_all = data_all.reshape(cfg.data_val.n_subs, -1, data_all.shape[-2], data_all.shape[-1])
    save_dir = os.path.join(cfg.data_val.data_dir,'ext_fea')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    np.save(save_dir+'/onesub_label2.npy',onesub_label2)
    
    
    
    if isinstance(cfg.finetune.valid_method, int):
        n_folds = cfg.finetune.valid_method
    elif cfg.train.valid_method == 'loo':
        n_folds = cfg.data_val.n_subs

    n_per = round(cfg.data_val.n_subs / n_folds)
    
    for fold in range(cfg.data_val.n_subs):
        log.info(f"sub:{fold}")
        
        train_subs = [fold]

        data2_train = data_all[train_subs] # (subs,vid*n_samples, 62, 1250)
        data2 = data2_train
        
        # print(data2[0,0])
        if cfg.ext_fea.normTrain:
            data2_fold = normTrain(data2,data2_train)
        else:
            log.info('no normTrain')
            data2_fold = data2
        # print(data2_fold[0,0])
        if cfg.ext_fea.use_pretrain:
            log.info('Use pretrain model:')
            data2_fold = data2_fold.reshape(-1, data2_fold.shape[-2], data2_fold.shape[-1])
            label2_fold = np.tile(onesub_label2, 1)
            foldset = SEEDV_Dataset(data2_fold, label2_fold)
            del data2_fold, label2_fold
            fold_loader = DataLoader(foldset, batch_size=cfg.ext_fea.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
            checkpoint =  os.path.join(cfg.log.logpath, cfg.log.proj_name, f'f{fold}_tuned.ckpt' if cfg.ext_fea.finetune else 'base.ckpt')
            checkpoint = glob.glob(checkpoint)[0]
            
            log.info('checkpoint load from: '+checkpoint)
            Extractor = MultiModel_PL.load_from_checkpoint(checkpoint_path=checkpoint, cfg=cfg)
            # Extractor.model.stratified = []
            Extractor.saveFea = True
            log.info('load model:'+checkpoint)

            # save dataset-wise cov_feature
            cov_1_mean, cov_2_mean = Extractor.cov_1_mean, Extractor.cov_2_mean
            save_batch_images(torch.stack([cov_1_mean, cov_2_mean]), 'cov_mean_extractor')
            save_path = os.path.join(save_dir,cfg.log.exp_name+f'_f{fold}_cov_'+cfg.ext_fea.mode+'.npy')
            np.save(save_path, torch.stack([cov_1_mean, cov_2_mean]).cpu())
            log.info('save covarience:'+save_path)

            trainer = pl.Trainer(accelerator='gpu', devices=cfg.train.gpus)
            pred = trainer.predict(Extractor, fold_loader)
            print(len)
            print(f'pred shape:{len(pred), pred[0].shape}')
            # save_img(pred[0].permute(0, 3, 1, 2).reshape(256*16, 960)[:1000, :], 'fea_before_norm.png')
            # data
            # pred = torch.stack(pred,dim=0)
            pred = torch.cat(pred, dim=0).cpu().numpy()
            log.debug(pred.shape)
            # pred = pred

            # max_fea = np.max(pred)
            # min_fea = np.min(pred)
            # print(max_fea,min_fea)
            # if np.isinf(pred).any():
            #     print("There are inf values in the array")

            fea = cal_fea(pred,cfg.ext_fea.mode)
            print(f'after cal_fea:{fea.shape}')
            # print('fea0:',fea[0])
            fea = fea.reshape(1,-1,fea.shape[-1])
            
        else:
            #data2_fold shape (n_subs,session*vid*n_samples, n_chans, n_pionts)
            log.info('Direct DE extraction:')
            n_subs, n_samples, n_chans, sfreqs = data2_fold.shape
            freqs = [[1,4], [4,8], [8,14], [14,30], [30,47]]
            de_data = np.zeros((n_subs, n_samples, n_chans, len(freqs)))
            n_samples2_onesub_cum = np.concatenate((np.array([0]), np.cumsum(n_samples2_onesub)))
            
            for idx, band in enumerate(freqs):
                for sub in range(n_subs):
                    log.debug(f'sub:{sub}')
                    for vid in tqdm(range(len(n_samples2_onesub)), desc=f'Direct DE Processing sub: {sub}', leave=False):
                        data_onevid = data2_fold[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]]
                        data_onevid = data_onevid.transpose(1,0,2)
                        data_onevid = data_onevid.reshape(data_onevid.shape[0],-1)
                        
                        data_video_filt = mne.filter.filter_data(data_onevid, sfreqs, l_freq=band[0], h_freq=band[1])
                        data_video_filt = data_video_filt.reshape(n_chans, -1, sfreqs)
                        de_onevid = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(data_video_filt, 2))).T
                        de_data[sub,  n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1], :, idx] = de_onevid
            fea = de_data.reshape(1, n_samples, -1)
        log.debug(fea.shape)    
        
        fea_train = fea[0]
        
        data_mean = np.mean(np.mean(fea_train, axis=1),axis=0)
        data_var = np.mean(np.var(fea_train, axis=1),axis=0)
        # print('fea_mean:',data_mean) 
        # print('fea_var:',data_var)
        if np.isinf(fea).any():
            log.warning("There are inf values in the array")
        else:
            log.info('no inf')
        if np.isnan(fea).any():
            log.warning("There are nan values in the array")
        else:
            log.info('no nan')
            
        # reorder
        if cfg.data_val.dataset_name == 'FACED':
            vid_order = video_order_load(cfg.data_val.n_vids)
            if cfg.data_val.n_class == 2:
                n_vids = 24
            elif cfg.data_val.n_class == 9:
                n_vids = 28
            vid_inds = np.arange(n_vids)
            fea, vid_play_order_new = reorder_vids_sepVideo(fea, vid_order, vid_inds, n_vids)


        n_sample_sum_sessions = np.sum(n_samples2_sessions,1)
        n_sample_sum_sessions_cum = np.concatenate((np.array([0]), np.cumsum(n_sample_sum_sessions)))
        # save_batch_images(fea[:, :1000, :], 'fea_before_norm')
        print(f'before norm:{fea.shape}')
        # fea_processed = np.zeros_like(fea)
        log.info('running norm:')
        for sub in range(1):
            log.debug(f'sub:{sub}')
            for s in  tqdm(range(len(n_sample_sum_sessions)), desc=f'running norm sub: {sub}', leave=False):
                fea[sub,n_sample_sum_sessions_cum[s]:n_sample_sum_sessions_cum[s+1]] = running_norm_onesubsession(
                                            fea[sub,n_sample_sum_sessions_cum[s]:n_sample_sum_sessions_cum[s+1]],
                                            data_mean,data_var,cfg.ext_fea.rn_decay)
                
        # print('rn:',fea[0,0])
        # save_batch_images(fea[:, :1000, :], 'fea_before_LDS')
        print(f'before LDS:{fea.shape}')
        if np.isinf(fea).any():
            log.warning("There are inf values in the array")
        else:
            log.info('no inf')
        if np.isnan(fea).any():
            log.warning("There are nan values in the array")
        else:
            log.info('no nan')
        save_batch_images(fea[:, :1000, :], 'fea_after_LDS')

        # order back
        if cfg.data_val.dataset_name == 'FACED':
            fea = reorder_vids_back(fea, len(vid_inds), vid_play_order_new)
        
        n_samples2_onesub_cum = np.concatenate((np.array([0]), np.cumsum(n_samples2_onesub)))
        # LDS
        # log.info('LDS:')
        # for sub in range(1):
        #     log.debug(f'sub:{sub}')
        #     for vid in tqdm(range(len(n_samples2_onesub)), desc=f'LDS Processing sub: {sub}', leave=False):
        #         fea[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]] = LDS_gpu(fea[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]])
        #     # print('LDS:',fea[sub,0])
        # fea = fea.reshape(-1,fea.shape[-1])
        # print(fea.shape)
        
        
        # (8.32145433e-18-8.31764020e-18)/np.sqrt(4.01888196e-40)
        
        # max_fea = np.max(fea)
        # min_fea = np.min(fea)
        # print(max_fea,min_fea)
        if np.isinf(fea).any():
            log.warning("There are inf values in the array")
        else:
            log.info('no inf')
        if np.isnan(fea).any():
            log.warning("There are nan values in the array")
        else:
            log.info('no nan')

        save_path = os.path.join(save_dir,cfg.log.exp_name+f'_sub{fold}_fea_{'pretrain_' if cfg.ext_fea.use_pretrain else ''}{cfg.ext_fea.mode if cfg.ext_fea.use_pretrain else 'DE'}.npy')
        # if not os.path.exists(cfg.ext_fea.save_dir):
        #     os.makedirs(cfg.ext_fea.save_dir)  
        np.save(save_path,fea)
        log.info(f'fea saved to {save_path}')
        
        if cfg.train.iftest :
            log.info('test mode!')
            break

    
def normTrain(data2,data2_train):
    log.info('normTrain')
    temp = np.transpose(data2_train,(0,1,3,2))
    temp = temp.reshape(-1,temp.shape[-1])
    data2_mean = np.mean(temp, axis=0)
    data2_var = np.var(temp, axis=0)
    data2_normed = (data2 - data2_mean.reshape(-1,1)) / np.sqrt(data2_var + 1e-5).reshape(-1,1)
    return data2_normed

def cal_fea(data,mode):
    if mode == 'de':
        # print(np.var(data, 3).squeeze()[0])
        fea = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(data, 3)) + 1.0).squeeze()
        fea[fea<-40] = -40
    elif mode == 'me':
        fea = np.mean(data, axis=3).squeeze()
    # print(fea.shape)
    # print(fea[0])
    return fea




if __name__ == '__main__':
    ext_fea()