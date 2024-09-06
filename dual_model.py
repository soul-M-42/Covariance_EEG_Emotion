import torch
import pytorch_lightning as pl
import torch.nn as nn
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MLLA_test import MLLA_BasicLayer
# os.environ["GEOMSTATS_BACKEND"] = 'pytorch' 
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import matplotlib.pyplot as plt
import itertools
import time

def logm(spd_matrices):
    eigvals, eigvecs = torch.linalg.eigh(spd_matrices)
    log_eigvals = torch.diag(torch.log(eigvals + 1.0))
    X_log = eigvecs @ log_eigvals @ eigvecs.transpose(-2, -1)
    return X_log

def div_std(x):
    x = x.clone() / torch.std(x.clone())
    return x

def mean_to_zero(x):
    x = x.clone() - torch.mean(x.clone())
    return x

# def log_euclidean_normalization(A):
#     save_img(A.detach().cpu(), 'A_BEFORE.png')
#     A = (A + A.T) / 2
#     A_log = logm(A)
#     A_log_normalized = (A_log - A_log.mean()) / A_log.std()
#     A_normalized = torch.linalg.matrix_exp(A_log_normalized)
#     save_img(A_normalized.detach().cpu(), 'A_AFTER.png')
#     return A_normalized

def save_img(data, filename='image_with_colorbar.png', cmap='viridis'):
    fig, ax = plt.subplots()
    cax = ax.imshow(data, cmap=cmap)
    fig.colorbar(cax)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def is_spd(mat, tol=1e-6):
    return True
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            if(mat[i][j] != mat[j][i]):
                print(mat[i][j], mat[j][i])
    return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real>=0).all())

class channelwiseEncoder(nn.Module):
    def __init__(self, standard_channels, n_filter):
        super().__init__()
        self.standard_channels = standard_channels
        self.n_channels = len(standard_channels)
        self.conv1d = nn.ModuleList([nn.Conv1d(1, n_filter, 8, 4, 1) for _ in range(self.n_channels)])

    def forward(self, x, x_channel_names):
        n_channels_data = len(x_channel_names)
        # assert n_channels_data == len(x_channel_names), "data channel not equal to channel_name dict length"
        outputs = []

        for i in range(n_channels_data):
            channel_data = x[:, :, i, :]
            # print(channel_data.shape)
            encoder_index = self.standard_channels.index(x_channel_names[i])
            out_channel_i = self.conv1d[encoder_index](channel_data)
            outputs.append(out_channel_i)
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.reshape((outputs.shape[0], -1, outputs.shape[-1]))
        # [batch, channel*n_filter, time]
        return outputs

class BiMap(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_shape, out_shape))
        # self.norm = nn.LayerNorm([out_shape, out_shape], elementwise_affine=False)
    def forward(self, x):
        out = self.W @ x @ self.W.transpose(-2, -1)
        out = (out - torch.mean(out, dim=0)) / torch.std(out, dim=0)
        # out = self.norm(out)
        return out
    
class ReEig(nn.Module):
    def __init__(self, eta):
        super().__init__()
        self.thresh = nn.Threshold(eta, eta)
    def forward(self, x):
        out = torch.zeros_like(x)
        for i, x_i in enumerate(x):
            eigvals, eigvecs = torch.linalg.eigh(x_i)
            eigvals = torch.diag(self.thresh(eigvals))
            out[i] = eigvecs @ eigvals @ eigvecs.transpose(-2, -1)
        return out
    
class ConvOut(nn.Module):
    # Input: a batch of SPD Covariance matrix [B, N, N]
    def __init__(self, in_shape):
        super().__init__()
        self.ReEig = ReEig(eta=0.005)
        self.bimap_1 = BiMap(in_shape=in_shape, out_shape=in_shape)
        self.bimap_2 = BiMap(in_shape=in_shape, out_shape=in_shape)

    def forward(self, x):
        x = self.bimap_1(x)
        x = self.ReEig(x)
        # x = self.bimap_2(x)
        # x = self.ReEig(x)
        out = x
        return out

class ConvOut_Euclidean(nn.Module):
    def __init__(self, out_channels=64, n_kernel=1):
        """
        SPD Convolution Layer with a pyramidal form.

        Args:
            out_channels (int): Number of output channels after each convolution layer.
            n_kernel (int): Number of convolution kernels (default is 1).
        """
        super(ConvOut_Euclidean, self).__init__()
        self.n_kernel = n_kernel
        self.out_channels = out_channels

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.ReLU()

        # Second convolutional layer with downsampling
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=3,
            stride=2,  # Downsampling by reducing the spatial size
            padding=1,
            bias=False
        )
        self.norm2 = nn.BatchNorm2d(out_channels * 2)
        self.activation2 = nn.ReLU()

        # Third convolutional layer with further downsampling
        self.conv3 = nn.Conv2d(
            in_channels=out_channels * 2,
            out_channels=out_channels * 4,
            kernel_size=3,
            stride=2,  # Further downsampling
            padding=1,
            bias=False
        )
        self.norm3 = nn.BatchNorm2d(out_channels * 4)
        self.activation3 = nn.ReLU()

        # Linear layer to merge channels into a single channel via a linear combination
        self.merge = nn.Linear(out_channels * 4, 1, bias=False)

    def forward(self, x):
        """
        Forward pass of SPD Convolution

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, N], where:
                              B - Batch size
                              N - SPD matrix dimension

        Returns:
            torch.Tensor: Output tensor of shape [B, final_N, final_N] after merging channels.
        """
        B, N, _ = x.shape

        # Unsqueeze to add channel dimension: [B, 1, N, N]
        x = x.unsqueeze(1)

        # First convolution layer
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)

        # Second convolution layer with downsampling
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)

        # Third convolution layer with further downsampling
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation3(x)

        # Permute to [B, final_N, final_N, out_channels * 4] to apply linear combination along channel dimension
        x = x.permute(0, 2, 3, 1)

        # Apply the linear layer to merge channels back to a single output [B, final_N, final_N, 1]
        x = self.merge(x)

        # Squeeze the last dimension to get the shape [B, final_N, final_N]
        x = x.squeeze(-1)

        return x
    

def to_patch(data, patch_size=50, stride=25):
    batchsize, timelen, dim = data.shape
    num_patches = (timelen - patch_size) // stride + 1
    patches = torch.zeros((batchsize, num_patches, patch_size)).to('cuda')
    for i in range(num_patches):
        start_idx = i * stride
        end_idx = start_idx + patch_size
        patches[:, i, :] = data[:, start_idx:end_idx, 0]
    return patches

class channel_MLLA(nn.Module):
    def __init__(self, standard_channels, patch_size, hidden_dim, out_dim, depth, patch_stride, drop_path, n_filter, filterLen):
        super().__init__()
        self.standard_channels = standard_channels
        self.n_channels = len(standard_channels)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.encoders = nn.ModuleList([MLLA_BasicLayer(
        #                             in_dim=patch_size, hidden_dim=hidden_dim, out_dim=out_dim,
        #                             depth=1, num_heads=8) 
        #                             for _ in range(self.n_channels)])
        self.encoder = MLLA_BasicLayer(
                                    in_dim=patch_size, hidden_dim=hidden_dim, out_dim=out_dim,
                                    depth=depth, num_heads=8, drop_path=drop_path, n_filter=n_filter, filterLen=filterLen)

    def forward(self, x, x_channel_names):
        n_channels_data = len(x_channel_names)
        # assert n_channels_data == len(x_channel_names), "data channel not equal to channel_name dict length"
        outputs = []

        for i in range(n_channels_data):
            channel_data = x[:, :, i, :]
            # print(channel_data.shape)
            # [Batch, channel, time]
            channel_data = channel_data.permute(0, 2, 1)
            channel_data = to_patch(channel_data, patch_size=self.patch_size, stride=self.patch_stride)
            # print(channel_data.shape)
            # [Batch, time(N), channel(C)]
            # encoder_index = self.standard_channels.index(x_channel_names[i])
            # out_channel_i = self.encoders[encoder_index](channel_data)
            out_channel_i = self.encoder(channel_data)
            channel_data = channel_data.permute(0, 2, 1)
            outputs.append(out_channel_i)
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.permute(0, 1, 3, 2)
        outputs = outputs.reshape((outputs.shape[0], -1, outputs.shape[-1]))
        # [batch, channel*n_filter, time]
        return outputs

class Channel_Alignment(nn.Module):
    def __init__(self, source_channel, target_channel):
        super().__init__()
        # self.A = nn.Parameter(torch.eye(source_channel) if source_channel ==  target_channel else torch.randn(source_channel, target_channel))
        self.A = nn.Parameter(torch.randn(source_channel, target_channel))
        # 30 * 60

    def forward(self, input):
        # input.shape = [batch, channel*n_filter, time']
        # 60(ch) * 30 * 1 * 2750
        # 30(ch) * 30 * 1 * 2750
        out = torch.permute(input, (0, 2, 1))
        # [batch, time', channel*n_filter]
        out = torch.matmul(out, self.A)
        return out



class DualModel_PL(pl.LightningModule):
    def __init__(self, cfg, dm) -> None:
        super().__init__()
        self.cfg = cfg
        self.channelwiseEncoder = channel_MLLA(
            standard_channels=cfg.data_1.channels, 
            patch_size=cfg.channel_encoder.patch_size,
            hidden_dim=cfg.channel_encoder.hidden_dim,
            out_dim=cfg.channel_encoder.out_dim,
            depth=cfg.channel_encoder.depth,
            patch_stride=cfg.channel_encoder.patch_stride,
            drop_path=cfg.channel_encoder.drop_path,
            n_filter=cfg.channel_encoder.n_filter,
            filterLen=cfg.channel_encoder.filterLen)
        # self.channelwiseEncoder = channelwiseEncoder(n_filter=cfg.channel_encoder.out_dim, standard_channels=cfg.data_1.channels)
        self.alignmentModule_1 = Channel_Alignment(cfg.data_1.n_channs*cfg.channel_encoder.out_dim, cfg.align.n_channel_uni)
        self.alignmentModule_2 = Channel_Alignment(cfg.data_2.n_channs*cfg.channel_encoder.out_dim, cfg.align.n_channel_uni)
        # self.decoder = ConvOut(in_shape=cfg.align.n_channel_uni)
        # self.decoder = ConvOut_Euclidean(out_channels=64)
        self.protos_1 = None
        self.protos_2 = None
        # self.protos_1 = self.init_proto_rand(dim=cfg.align.n_channel_uni, n_class=cfg.data_1.n_class)
        # self.protos_2 = self.init_proto_rand(dim=cfg.align.n_channel_uni, n_class=cfg.data_2.n_class)
        self.lr = cfg.train.lr
        self.wd = cfg.train.wd
        self.max_epochs = cfg.train.max_epochs
        self.restart_times = cfg.train.restart_times
        self.is_logger = cfg.log.is_logger
        self.align_factor = cfg.align.factor
        self.tts_1 = None
        self.tts_2 = None
        # self.dm = dm
        # self.train_dataset = dm.train_dataset
        # self.train_set_1 = self.train_dataset.dataset_a
        # self.train_set_2 = self.train_dataset.dataset_b
    

    def cov_mat(self, data):
        batch_size, t, num_channels = data.shape
        data = data - data.mean(dim=1, keepdim=True)
        # for data_i in data_centered:
        #     # save_img(data_i.detach().cpu(), 'data_mat.png')
        #     # time.sleep(1)
        #     print(data_i.shape)
        cov_matrices = torch.matmul(data.transpose(1, 2), data) / (t - 1)
            # cov_matrices[i] = log_euclidean_normalization(cov_matrices[i])
        # print(cov_matrices.shape)
        # mean = cov_matrices.mean(dim=0)
        # std = cov_matrices.std(dim=0)
        # cov_matrices = (cov_matrices - mean) / std
        # for cov in cov_matrices:
        #     if(not is_spd(cov)):
        #         raise('not sym cov')

        return cov_matrices


    

    def frechet_mean(self, cov_matrices, tol=1e-5, max_iter=100):
        centroid = torch.mean(cov_matrices, axis=0)
        for i in range(max_iter):
            prev_centroid = centroid.clone()
            weights = torch.zeros(len(cov_matrices))
            for j, cov_matrix in enumerate(cov_matrices):
                weights[j] = 1.0 / self.frobenius_distance(cov_matrix, centroid)
            weights /= torch.sum(weights)
            centroid = torch.zeros_like(centroid)
            for j, cov_matrix in enumerate(cov_matrices):
                centroid += weights[j] * cov_matrix
            # 检查是否达到终止条件
            total_dis = 0
            for j, cov_matrix in enumerate(cov_matrices):
                total_dis += self.frobenius_distance(cov_matrix, centroid)
            delta = self.frobenius_distance(centroid, prev_centroid)
            # print(f'Ite {i} total_dis {total_dis / cov_matrices.shape[0]} delta {delta}')
            if self.frobenius_distance(centroid, prev_centroid) < tol:
                break
        return centroid

    # CDA for Cross_dataset Alignment
    def CDA_loss(self, cov_1, cov_2):
        dis = 0
        cov_1 = cov_1.reshape(2, -1, self.cfg.align.n_channel_uni, self.cfg.align.n_channel_uni)
        cov_2 = cov_2.reshape(2, -1, self.cfg.align.n_channel_uni, self.cfg.align.n_channel_uni)
        ind_cen = []
        ind_cen.append(self.get_ind_cen(cov_1[0]))
        ind_cen.append(self.get_ind_cen(cov_1[1]))
        ind_cen.append(self.get_ind_cen(cov_2[0]))
        ind_cen.append(self.get_ind_cen(cov_2[1]))
        for i in range(len(ind_cen)):
            for j in range(i+1, len(ind_cen)):
                cen_i = ind_cen[i]
                cen_j = ind_cen[j]
                dis = dis + self.frobenius_distance(cen_i, cen_j)
        loss = torch.log(dis + 1.0)
        return loss

    def frobenius_distance(self, matrix_a, matrix_b):
        # 计算Frobenius距离
        if matrix_a.shape != matrix_b.shape:
            raise ValueError("Must have same shape")
        if type(matrix_a) == torch.Tensor:
            if self.cfg.align.to_riem:
                distance = torch.linalg.norm(logm(matrix_a) - logm(matrix_b), 'fro')
                distance = distance ** 2
            else:
                distance = torch.linalg.norm(matrix_a - matrix_b, 'fro')
                distance = distance ** 2
        return distance

    def top1_accuracy(self, dis_mat, labels):
        # Get the predicted classes (indices of the 到原型最小距离 in each row)
        predicted_classes = torch.argmax(dis_mat, dim=1)
        
        # Flatten labels to match predicted_classes shape
        labels = labels.view(-1)
        
        # Calculate the number of correct predictions
        correct_predictions = (predicted_classes == labels).sum().item()
        
        # Calculate accuracy
        accuracy = correct_predictions / labels.size(0)
        
        return accuracy

    def init_proto_rand(self, dim, n_class):
        T_init = 64
        prototype_init = torch.zeros((n_class, dim, dim))
        for i in range(n_class):
            prototype_init[i] = div_std(self.cov_mat(torch.randn(1, T_init, dim)))
        return nn.Parameter(prototype_init)

    def loss_pair(self, cov, emo_label):
        n_vid = cov.shape[0] // 2
        similarity_matrix = torch.zeros((cov.shape[0], cov.shape[0]))
        labels = torch.cat([torch.arange(n_vid) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        for i in range(cov.shape[0]):
            for j in range(i+1, cov.shape[0]):
                cov_i = cov[i]
                cov_j = cov[j]
                similarity_matrix[i][j] = -self.frobenius_distance(cov_i, cov_j)
                similarity_matrix[j][i] = similarity_matrix[i][j]
        


        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        similarity_matrix = mean_to_zero(similarity_matrix)
        similarity_matrix = div_std(similarity_matrix)

        save_img(similarity_matrix.detach().cpu(), 'dis_pair.png')
        save_img(labels.detach().cpu(), 'label_pair.png')

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([negatives, positives], dim=1)
        labels = torch.ones(logits.shape[0], dtype=torch.long)*(logits.shape[1]-1)
        CEloss = nn.CrossEntropyLoss()
        pair_loss = CEloss(logits, labels)
        print(f'Contrasive_loss:{pair_loss}')
        return pair_loss

    def loss_proto(self, cov, label, protos):
        # print(cov.shape)
        # print(label.shape)
        batch_size = cov.shape[0]
        loss = 0
        dis_mat = torch.zeros((batch_size, protos.shape[0]))
        for i, cov_i in enumerate(cov):
            # save_img(cov_i.detach().cpu(), 'cov_i.png')
            for j, pro_j in enumerate(protos):
                # pro_j = div_std(pro_j)
                # print(is_spd(cov_i))
                # print(is_spd(pro_j))
                dis_mat[i][j] = -self.frobenius_distance(cov_i, pro_j)
                # save_img(pro_j.detach().cpu(), 'pro_j.png')
                # print(-dis_mat[i][j])
            dis_mat[i] = div_std(dis_mat[i])
            dis_mat[i] = mean_to_zero(dis_mat[i])
            dis_mat[i] = F.softmax(dis_mat[i])
            
        
        save_img(dis_mat.detach().cpu(), 'dis_pro.png')
        num_classes = label.max().item() + 1
        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)
        # save_img(label_onehot.detach().cpu(), 'label_pro.png')

        # print(dis_mat)
        acc = self.top1_accuracy(dis_mat, label)
        # dis_mat = F.log_softmax(dis_mat, dim=1)
        CEloss = nn.CrossEntropyLoss()
        # NLLLoss = nn.NLLLoss()
        # loss = NLLLoss(dis_mat, label)
        loss_ce = CEloss(dis_mat, label)
        # print(loss)
        # print(loss_ce)
        # print(loss)

        return loss_ce, acc

    def extract_leading_eig_vector(self, data_dim, data, device='gpu'):
        geom_backend = torch if device == 'gpu' else np
        # data_dim = 4
        matrix_data = geom_backend.zeros([data.shape[0],data_dim,data_dim])
        
        for specimen in range(data.shape[0]):
            this_specimen_data = data[specimen,:]
            this_specimen_matrix = geom_backend.zeros([data_dim,data_dim])
            # for 
            # this_specimen_matrix[i,j]
            for row in range(data_dim):
                
                col_data = geom_backend.concatenate((geom_backend.zeros(row),this_specimen_data[int((row*(data_dim*2-row+1))/2):int(((row*(data_dim*2-row+1))/2))+data_dim-row]))
                
                this_specimen_matrix[row,:] = col_data
                
            this_specimen_matrix = (this_specimen_matrix - geom_backend.eye(data_dim,data_dim) * this_specimen_matrix).T +this_specimen_matrix
            
            matrix_data[specimen,:,:] = this_specimen_matrix
            
            # reshape_matrix_data
        return matrix_data   

    def get_ind_cen(self, mat):
        mat = torch.squeeze(mat)
        mat = self.frechet_mean(mat)
        return mat
    
    def forward(self, batch, channel_names):
        feature = self.channelwiseEncoder(batch, channel_names)
        return feature 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.max_epochs, gamma=0.8, last_epoch=-1, verbose=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.max_epochs // self.restart_times, eta_min=0,last_epoch=-1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    # remain to be implemented
    def training_step(self, batch, batch_idx):
        x_1, y_1, x_2, y_2 = batch
        x_1 = x_1[0]
        x_2 = x_2[0]
        y_1 = y_1[0]
        y_2 = y_2[0]
        fea_1 = self.forward(x_1, self.cfg.data_1.channels)
        fea_2 = self.forward(x_2, self.cfg.data_2.channels)
        
        fea_1 = self.alignmentModule_1(fea_1)
        fea_2 = self.alignmentModule_2(fea_2)
        cov_1 = self.cov_mat(fea_1)
        cov_2 = self.cov_mat(fea_2)
        
        for i in range(len(cov_1)):
            cov_1[i] = div_std(cov_1[i])
        for i in range(len(cov_2)):
            cov_2[i] = div_std(cov_2[i])
        # cov_1 = self.decoder(cov_1)
        # cov_2 = self.decoder(cov_2)
        # if self.cfg.align.to_riem:
        #     cov_1, tts_1 = self.cov_to_riem(cov_1, self.tts_1, 'train', device=self.cfg.align.device)
        #     cov_2, tts_2 = self.cov_to_riem(cov_2, self.tts_2, 'train', device=self.cfg.align.device)
        #     self.tts_1 = tts_1
        #     self.tts_2 = tts_2

        # print(cov_1.shape, cov_2.shape)
        # print(y_1, y_2)
        loss_1 = 0
        loss_2 = 0
        if self.protos_1 is None:
            self.protos_1 = self.init_proto_rand(dim=cov_1.shape[1], n_class=self.cfg.data_1.n_class)
            self.protos_2 = self.init_proto_rand(dim=cov_2.shape[1], n_class=self.cfg.data_2.n_class)
        loss_class_1, acc_1 = self.loss_proto(cov_1, y_1, self.protos_1)
        loss_class_2, acc_2 = self.loss_proto(cov_2, y_2, self.protos_2)

        loss_pair_1 = self.loss_pair(cov_1, y_1)
        loss_pair_2 = self.loss_pair(cov_2, y_2)

        loss_1 = loss_1 + loss_class_1
        loss_2 = loss_2 + loss_class_2

        if self.cfg.align.clisa_loss:
            loss_1 = loss_1 + loss_pair_1
            loss_2 = loss_2 + loss_pair_2

        print(f'train/loss_class_1={loss_class_1},train/loss_class_2={loss_class_2},train/acc_1={acc_1},train/acc_2={acc_2}')
        
        # Check grad 
        check_grad = 0
        if check_grad:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    if 'encoder' not in name:
                        grad_norm = param.grad.data.norm(2).item()
                        print(f'grad_norm_{name}', grad_norm)
                    else:
                        grad_norm = param.grad.data.norm(2).item()
                        print(f'grad_norm_{name}', grad_norm)
                        pass
        # print('\n')
        
        # print('loss_class_1/train,',loss_class_1, 
        #             '\nloss_class_2/train ',loss_class_2)
        self.log_dict({
                    'loss_class_1/train': loss_class_1, 
                    'loss_class_2/train': loss_class_2, 
                    'loss_pair_1/train': loss_pair_1, 
                    'loss_pair_2/train': loss_pair_2, 
                    'acc_1/train': acc_1, 
                    'acc_2/train': acc_2, 
                    'lr': self.optimizers().param_groups[-1]['lr']
                    },
                    logger=self.is_logger,
                    on_step=False, on_epoch=True, prog_bar=True)
        # fea.shape = [channel, n_filter, time']
        # self.criterion.to(data.device)   # put it in the loss function
        # for name, param in self.channelwiseEncoder.named_parameters():
        #     if param.requires_grad:
        #         print(f"Gradient of {name}: {param.grad}")
        loss = loss_1 + loss_2
        if self.cfg.align.align_loss:
            loss_Align = self.align_factor * self.CDA_loss(cov_1, cov_2)
            print(f'loss_Align={loss_Align}')
            loss = loss + loss_Align
            self.log_dict({
                    'loss_align/train': loss_Align, 
                    },
                    logger=self.is_logger,
                    on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_1, y_1, x_2, y_2 = batch
        x_1 = x_1[0]
        x_2 = x_2[0]
        y_1 = y_1[0]
        y_2 = y_2[0]
        # print(x_1.shape, x_2.shape, y_1.shape, y_2.shape)
        fea_1 = self.forward(x_1, self.cfg.data_1.channels)
        fea_2 = self.forward(x_2, self.cfg.data_2.channels)
        fea_1 = self.alignmentModule_1(fea_1)
        fea_2 = self.alignmentModule_2(fea_2)
        cov_1 = self.cov_mat(fea_1)
        cov_2 = self.cov_mat(fea_2)
        # cov_1 = self.decoder(cov_1)
        # cov_2 = self.decoder(cov_2)
        
        for i in range(len(cov_1)):
            cov_1[i] = div_std(cov_1[i])
        for i in range(len(cov_2)):
            cov_2[i] = div_std(cov_2[i])
            
        loss_1 = 0
        loss_2 = 0
        
        # if self.cfg.align.to_riem:
        #     if self.tts_1 is not None:
        #         cov_1, _ = self.cov_to_riem(cov_1, self.tts_1, 'val', device=self.cfg.align.device)
        #         cov_2, _ = self.cov_to_riem(cov_2, self.tts_2, 'val', device=self.cfg.align.device)

        # print(loss_Align)
        # print(y_1, y_2)
        if self.protos_1 is None:
            self.protos_1 = self.init_proto_rand(dim=cov_1.shape[1], n_class=self.cfg.data_1.n_class)
            self.protos_2 = self.init_proto_rand(dim=cov_2.shape[1], n_class=self.cfg.data_2.n_class)

        loss_class_1, acc_1 = self.loss_proto(cov_1, y_1, self.protos_1)
        loss_class_2, acc_2 = self.loss_proto(cov_2, y_2, self.protos_2)

        loss_pair_1 = self.loss_pair(cov_1, y_1)
        loss_pair_2 = self.loss_pair(cov_2, y_2)

        loss_1 = loss_1 + loss_class_1
        loss_2 = loss_2 + loss_class_2
        
        
        if self.cfg.align.clisa_loss:
            loss_1 = loss_1 + loss_pair_1
            loss_2 = loss_2 + loss_pair_2

        # print(f'test/loss_class_1={loss_class_1},test/loss_class_2={loss_class_2},test/acc_1={acc_1},test/acc_2={acc_2}')
        self.log_dict({
                    'loss_class_1/val': loss_class_1, 
                    'loss_class_2/val': loss_class_2, 
                    'loss_pair_1/val': loss_pair_1, 
                    'loss_pair_2/val': loss_pair_2, 
                    'acc_1/val': acc_1, 
                    'acc_2/val': acc_2, 
                    'lr/val': self.optimizers().param_groups[-1]['lr']
                    },
                    logger=self.is_logger,
                    on_step=False, on_epoch=True, prog_bar=True)
        # fea.shape = [channel, n_filter, time']
        # self.criterion.to(data.device)   # put it in the loss function
        loss = loss_1 + loss_2
        if self.cfg.align.align_loss:
            loss_Align = self.align_factor * self.CDA_loss(cov_1, cov_2)
            print(f'loss_Align={loss_Align}')
            loss = loss + loss_Align
            self.log_dict({
                    'loss_align/val': loss_Align, 
                    },
                    logger=self.is_logger,
                    on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        data_1, data_2 = batch
        x_1, y_1 = data_1
        x_2, y_2 = data_2
        fea_1 = self.forward(x_1, self.cfg.data_1.channels)
        fea_2 = self.forward(x_2, self.cfg.data_2.channels)
        return fea_1