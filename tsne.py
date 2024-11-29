import hydra
from omegaconf import DictConfig
import numpy as np
import os
from data.dataset import PDataset
from torch.utils.data import DataLoader
import torch
import logging
from sklearn.manifold import TSNE
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

log = logging.getLogger(__name__)

@hydra.main(config_path="cfgs_multi", config_name="config_multi", version_base="1.3")
def visualize_tsne(cfg: DictConfig) -> None:
    
    n_subs = cfg.data_val.n_subs
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    if isinstance(cfg.train.valid_method, int):
        n_folds = cfg.train.valid_method
    elif cfg.train.valid_method == 'loo':
        n_folds = cfg.data_val.n_subs

    n_per = round(cfg.data_val.n_subs / n_folds)
    
    for fold in range(0, n_folds):
        log.info(f"fold:{fold}")
        if n_folds == 1:
            val_subs = []
        elif fold < n_folds - 1:
            val_subs = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_subs = np.arange(n_per * fold, cfg.data_val.n_subs)            
        train_subs = list(set(np.arange(cfg.data_val.n_subs)) - set(val_subs))
        
        if cfg.train.reverse:
            train_subs, val_subs = val_subs, train_subs
        log.info(f'finetune_subs:{train_subs}')
        log.info(f'val_subs:{val_subs}')
        
        save_dir = os.path.join(cfg.data_val.data_dir, 'ext_fea')
        save_path = os.path.join(save_dir, cfg.log.exp_name + f'_f{fold}_fea_{'pretrain_' if cfg.ext_fea.use_pretrain else ''}{cfg.ext_fea.mode if cfg.ext_fea.use_pretrain else 'DE'}.npy')
        data2 = np.load(save_path)
        log.info('data2 load from: ' + save_path)
        
        if np.isnan(data2).any():
            log.warning('nan in data2')
            data2 = np.where(np.isnan(data2), 0, data2)
        
        fea_dim = data2.shape[-1]
        data2 = data2.reshape(cfg.data_val.n_subs, -1, fea_dim)
        print(data2.shape)
        n_sample = data2.shape[1]
        
        onesub_label2 = np.load(save_dir + '/onesub_label2.npy')
        print(onesub_label2.shape)
        labels2_train = np.tile(onesub_label2, len(train_subs))
        labels2_val = np.tile(onesub_label2, len(val_subs))

        # Prepare data for t-SNE visualization
        features, labels, subjects = [], [], []
        clip = 400
        mask = np.arange(n_sample)
        random.shuffle(mask)
        mask = mask[:clip]
        for sub_idx in range(3):
            features.append(data2[sub_idx][mask])
            labels.append(onesub_label2[mask])
            subjects.append(np.full(clip, sub_idx))
        
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        subjects = np.concatenate(subjects, axis=0)

        print(features.shape)
        print(labels.shape)
        print(subjects.shape)
        
        # Perform t-SNE visualization
        log.info("Performing t-SNE...")
        tsne = TSNE(n_components=2, random_state=cfg.seed, n_iter=1000, perplexity=200)
        features_2d = tsne.fit_transform(features)

        # Plot the t-SNE results colored by subjects
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=subjects, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label='Source Subjects')
        plt.title('t-SNE Visualization of Feature Distribution (Colored by Source Subjects)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True)
        plt.savefig(os.path.join('./tsne_out', f'tsne_fold_{fold}_subject.png'))
        plt.show()
        
        # Plot the t-SNE results colored by labels
        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)
        cmap = plt.get_cmap('hsv', num_labels)  # Use a color map with distinct colors for each label
        norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, num_labels + 0.5, 1), ncolors=num_labels)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=cmap, norm=norm, alpha=0.7)
        cbar = plt.colorbar(scatter, ticks=np.arange(num_labels))
        cbar.set_label('Labels')
        cbar.set_ticks(unique_labels)
        cbar.set_ticklabels(unique_labels)
        plt.title('t-SNE Visualization of Feature Distribution (Colored by Labels)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True)
        plt.savefig(os.path.join('./tsne_out', f'tsne_fold_{fold}_label.png'))
        plt.show()
        
        if cfg.train.iftest:
            break

if __name__ == '__main__':
    visualize_tsne()