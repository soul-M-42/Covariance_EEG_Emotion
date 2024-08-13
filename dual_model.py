import torch
import pytorch_lightning as pl
import torch.nn as nn
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MLLA_test import MLLA_BasicLayer
os.environ["GEOMSTATS_BACKEND"] = 'pytorch' 
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import geomstats._backend as gs
from geomstats.learning.preprocessing import ToTangentSpace
from geomstats.geometry.spd_matrices import SPDAffineMetric
from geomstats.geometry.spd_matrices import SPDMatrices
import matplotlib.pyplot as plt

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
    def __init__(self, standard_channels, patch_size, hidden_dim, out_dim, patch_stride):
        super().__init__()
        self.standard_channels = standard_channels
        self.n_channels = len(standard_channels)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.encoders = nn.ModuleList([MLLA_BasicLayer(
                                    in_dim=patch_size, hidden_dim=hidden_dim, out_dim=out_dim,
                                    depth=1, num_heads=8) 
                                    for _ in range(self.n_channels)])

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
            encoder_index = self.standard_channels.index(x_channel_names[i])
            out_channel_i = self.encoders[encoder_index](channel_data)
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
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.channelwiseEncoder = channel_MLLA(
            standard_channels=cfg.data_1.channels, 
            patch_size=cfg.channel_encoder.patch_size,
            hidden_dim=cfg.channel_encoder.hidden_dim,
            out_dim=cfg.channel_encoder.out_dim,
            patch_stride=cfg.channel_encoder.patch_stride)
        # self.channelwiseEncoder = channelwiseEncoder(n_filter=cfg.channel_encoder.out_dim, standard_channels=cfg.data_1.channels)
        self.alignmentModule_1 = Channel_Alignment(cfg.data_1.n_channs*cfg.channel_encoder.out_dim, cfg.align.n_channel_uni)
        self.alignmentModule_2 = Channel_Alignment(cfg.data_2.n_channs*cfg.channel_encoder.out_dim, cfg.align.n_channel_uni)
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
    
    def show_and_save_img(self, data, name):
        if isinstance(data, torch.Tensor) and data.is_cuda:
            data = data.detach().cpu().numpy()
        elif isinstance(data, torch.Tensor):
            data = data.numpy()
        plt.imshow(data, cmap='hot', interpolation='nearest')
        # plt.colorbar()
        plt.savefig(f'/home/Covariance_EEG_Emotion/{name}.png')

    def cov_mat(self, data):
        batch_size, t, num_channels = data.shape
        data_centered = data - data.mean(dim=1, keepdim=True)
        cov_matrices = torch.matmul(data_centered.transpose(1, 2), data_centered) / (t - 1)
        # print(cov_matrices.shape)
        # mean = cov_matrices.mean(dim=0)
        # std = cov_matrices.std(dim=0)
        # cov_matrices = (cov_matrices - mean) / std

        return cov_matrices

    def logm(self, spd_matrices):
        if not self.is_spd(spd_matrices):
            print('FUCK')
        # return spd_matrices

        # u, s, v = torch.linalg.svd(spd_matrices)
        # log_cov=torch.matmul(torch.matmul(u, torch.diag_embed(torch.log(s + 1.0))), v)
        # return log_cov
        eigvals, eigvecs = torch.linalg.eigh(spd_matrices)
        log_eigvals = torch.diag(torch.log(eigvals))
        X_log = eigvecs @ log_eigvals @ eigvecs.transpose(-2, -1)
        return X_log

    def is_spd(self, matrix, tol=1e-6):
        """
        检测一个矩阵是否是对称正定（SPD）的
        :param matrix: 输入矩阵
        :param tol: 数值容忍度，用于检查对称性
        :return: 如果矩阵是对称正定的，返回 True；否则返回 False
        """
        # 检查矩阵是否对称
        if not torch.allclose(matrix, matrix.t(), atol=tol):
            return False
        
        # 计算特征值
        eigvals = torch.linalg.eigvalsh(matrix)
        
        # 检查所有特征值是否为正
        if torch.any(eigvals <= 0):
            return False
        
        return True

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
        centroid_1 = self.frechet_mean(cov_1)
        centroid_2 = self.frechet_mean(cov_2)
        # print(is_spd(centroid_1), is_spd(centroid_2))
        # print(is_spd(centroid_1), is_spd(centroid_2))
        dis = self.frobenius_distance(centroid_1, centroid_2)
        loss = torch.log(dis + 1.0)
        return loss

    def frobenius_distance(self, matrix_a, matrix_b):
        # 计算Frobenius距离
        if matrix_a.shape != matrix_b.shape:
            raise ValueError("Must have same shape")
        if type(matrix_a) == torch.Tensor:
            if self.cfg.align.to_riem:
                distance = torch.linalg.norm(self.logm(matrix_a) - self.logm(matrix_b), 'fro')
                distance = distance ** 2
            else:
                distance = torch.linalg.norm(matrix_a - matrix_b, 'fro')
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

    def init_proto(self, cov, y, n_class):
        class_mean_cov = torch.zeros((n_class, cov.size(1), cov.size(2)))
        print(y, y.shape)
        for i in range(n_class):
            class_cov = cov[y == i]
            class_mean_cov[i] = self.frechet_mean(class_cov)
        return nn.Parameter(class_mean_cov)

    def init_proto_rand(self, dim, n_class):
        return nn.Parameter(torch.randn((n_class, dim, dim)))

    def loss_proto(self, cov, label, protos):
        # print(cov.shape)
        # print(label.shape)
        # print(protos.shape)
        batch_size = cov.shape[0]
        loss = 0
        dis_mat = torch.zeros((batch_size, protos.shape[0]))
        for i, cov_i in enumerate(cov):
            for j, pro_j in enumerate(protos):
                # print(is_spd(cov_i), is_spd(pro_j))
                dis_mat[i][j] = -self.frobenius_distance(cov_i, pro_j)
        # print(dis_mat)
        acc = self.top1_accuracy(dis_mat, label)
        dis_mat = F.log_softmax(dis_mat, dim=-1)
        NLLLoss = nn.NLLLoss()
        # print(dis_mat)
        loss = NLLLoss(dis_mat, label)
        # print(loss)

        return loss, acc

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

    def cov_to_riem(self, cov, tts, phase, device='gpu'):
        n_channel = cov.shape[1]
        manifold_1 = SPDMatrices(n_channel, equip=False)
        manifold_1.equip_with_metric(SPDAffineMetric)
        if phase == 'train':
            if tts is None:
                tts = ToTangentSpace(space=manifold_1)
            # for cov_i in cov:
            #     print(manifold_1.belongs(cov_i))
            if(device == 'cpu'):
                tts.fit(cov.cpu().detach().numpy())
            elif(device == 'gpu'):
                tts.fit(cov)
            else:
                raise("WTF device r u using")
        if device == 'cpu':
            cov = tts.transform(cov.cpu().detach().numpy())
        elif device == 'gpu':
            cov = tts.transform(cov)
        
        cov = self.extract_leading_eig_vector(n_channel,cov,device=device)
        # cov = torch.tensor(cov)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # cov = cov.to(device)
        return cov, tts
    
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
        data_1, data_2 = batch
        x_1, y_1 = data_1
        x_2, y_2 = data_2
        fea_1 = self.forward(x_1, self.cfg.data_1.channels)
        fea_2 = self.forward(x_2, self.cfg.data_2.channels)
        
        fea_1 = self.alignmentModule_1(fea_1)
        fea_2 = self.alignmentModule_2(fea_2)
        cov_1 = self.cov_mat(fea_1)
        cov_2 = self.cov_mat(fea_2)
        # if self.cfg.align.to_riem:
        #     cov_1, tts_1 = self.cov_to_riem(cov_1, self.tts_1, 'train', device=self.cfg.align.device)
        #     cov_2, tts_2 = self.cov_to_riem(cov_2, self.tts_2, 'train', device=self.cfg.align.device)
        #     self.tts_1 = tts_1
        #     self.tts_2 = tts_2

        # print(cov_1.shape, cov_2.shape)
        # print(y_1, y_2)
        if self.protos_1 is None:
            self.protos_1 = self.init_proto(cov_1, y_1, self.cfg.data_1.n_class)
            self.protos_2 = self.init_proto(cov_2, y_2, self.cfg.data_2.n_class)
        loss_class_1, acc_1 = self.loss_proto(cov_1, y_1, self.protos_1)
        loss_class_2, acc_2 = self.loss_proto(cov_2, y_2, self.protos_2)
        print(f'train/loss_class_1={loss_class_1},train/loss_class_2={loss_class_2},train/acc_1={acc_1},train/acc_2={acc_2}')
        
        # Check grad 
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         # if 'encoder' not in name:
        #             grad_norm = param.grad.data.norm(2).item()
        #             print(f'grad_norm_{name}', grad_norm)
        # print('\n')
        
        # print('loss_class_1/train,',loss_class_1, 
        #             '\nloss_class_2/train ',loss_class_2)
        self.log_dict({
                    'loss_class_1/train': loss_class_1, 
                    'loss_class_2/train': loss_class_2, 
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
        loss = loss_class_1 + loss_class_2
        if self.cfg.align.align_loss:
            loss_Align = self.align_factor * self.CDA_loss(cov_1, cov_2)
            loss = loss + loss_Align
            self.log_dict({
                    'loss_align/train': loss_Align, 
                    },
                    logger=self.is_logger,
                    on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        data_1, data_2 = batch
        x_1, y_1 = data_1
        x_2, y_2 = data_2
        fea_1 = self.forward(x_1, self.cfg.data_1.channels)
        fea_2 = self.forward(x_2, self.cfg.data_2.channels)
        fea_1 = self.alignmentModule_1(fea_1)
        fea_2 = self.alignmentModule_2(fea_2)
        cov_1 = self.cov_mat(fea_1)
        cov_2 = self.cov_mat(fea_2)
        
        # if self.cfg.align.to_riem:
        #     if self.tts_1 is not None:
        #         cov_1, _ = self.cov_to_riem(cov_1, self.tts_1, 'val', device=self.cfg.align.device)
        #         cov_2, _ = self.cov_to_riem(cov_2, self.tts_2, 'val', device=self.cfg.align.device)

        # print(loss_Align)
        # print(y_1, y_2)
        if self.protos_1 is None:
            return None
        loss_class_1, acc_1 = self.loss_proto(cov_1, y_1, self.protos_1)
        loss_class_2, acc_2 = self.loss_proto(cov_2, y_2, self.protos_2)
        # print(f'test/loss_class_1={loss_class_1},test/loss_class_2={loss_class_2},test/acc_1={acc_1},test/acc_2={acc_2}')
        self.log_dict({
                    'loss_class_1/val': loss_class_1, 
                    'loss_class_2/val': loss_class_2, 
                    'acc_1/val': acc_1, 
                    'acc_2/val': acc_2, 
                    'lr/val': self.optimizers().param_groups[-1]['lr']
                    },
                    logger=self.is_logger,
                    on_step=False, on_epoch=True, prog_bar=True)
        # fea.shape = [channel, n_filter, time']
        # self.criterion.to(data.device)   # put it in the loss function
        loss = loss_class_1 + loss_class_2
        if self.cfg.align.align_loss:
            loss_Align = self.align_factor * self.CDA_loss(cov_1, cov_2)
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