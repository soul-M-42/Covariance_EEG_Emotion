import torch
# import torch.nn.functional as F
import torch.nn as nn
# import numpy as np
# import time
from riemannian_utils import get_centroid, frobenius_distance

def euclidean_distance_loss(cov_matrices, baseline_cov):
    """
    计算欧几里得距离下的损失函数。
    
    :param cov_matrices: 形状为 [batchsize, 30, 30] 的协方差矩阵
    :param baseline_cov: 形状为 [30, 30] 的基线协方差矩阵
    :return: 欧几里得距离损失
    """
    # 扩展基线协方差矩阵以匹配批次维度
    baseline_cov_expanded = baseline_cov.unsqueeze(0).expand_as(cov_matrices)
    # 计算欧几里得距离损失
    loss = torch.mean(torch.norm(cov_matrices - baseline_cov_expanded, dim=(1, 2)))
    return loss

def riemannian_distance_loss(cov_matrices, baseline_cov):
    loss = 0
    for cov_matrix in cov_matrices:
        loss += frobenius_distance(cov_matrix, baseline_cov)
    return loss / cov_matrices.shape[0]

def covariance_matrices_divergence(cov_matrices):
    """
    计算一组协方差矩阵的Frobenius散度

    参数:
    cov_matrices (numpy.ndarray): 形状为 (batchsize, n, n) 的协方差矩阵列表

    返回:
    float: 矩阵列表的散度
    """
    batchsize, n, n = cov_matrices.shape
    mean_cov_matrix = torch.mean(cov_matrices, axis=0)
    divergence = 0.0

    for cov_matrix in cov_matrices:
        divergence += frobenius_distance(cov_matrix, mean_cov_matrix) ** 2

    divergence = torch.sqrt(divergence / batchsize)
    return divergence

class CovLoss(nn.Module):
    def __init__(self, cfg):
        super(CovLoss, self).__init__()
        self.cfg = cfg
        self.centroid = []
    def forward(self, data, Linear_Layer, source_centroid, loss_type):
        # loss_type: 'div' for minimize SPD divergance, or 'centr' for SPD alignment
        # print(data.shape)
        # [batchzise, 1, n_channel_source, n_time]
        # print(Linear_Layer.shape)
        # [batchzise, n_channel_target]
        data_tensor = data.squeeze(1)  # 现在形状为 (batchzise, n_channel, n_time)
        batch_size = data_tensor.shape[0]
        num_features = data_tensor.shape[1]
        cov_matrices = torch.zeros((batch_size, Linear_Layer.shape[0], Linear_Layer.shape[0]))
        for i in range(batch_size):
            cov_i = torch.cov(data_tensor[i])
            # print(cov_i.shape)
            cov_i = torch.matmul(Linear_Layer, cov_i)
            cov_i = torch.matmul(cov_i, Linear_Layer.T)
            cov_matrices[i] = cov_i
        # [batchzise, n_channel_target, n_channel_target]
        if loss_type == 'div':
            SPD_divergance = covariance_matrices_divergence(cov_matrices)
            return SPD_divergance
        if self.cfg.train.alignment == 'Euclidean':
            return euclidean_distance_loss(cov_matrices, source_centroid), None
        if self.cfg.train.alignment == 'Riemannian':
            return torch.pow(riemannian_distance_loss(cov_matrices, source_centroid), 2) / (4 * source_centroid.shape[0] * source_centroid.shape[0]), None
        if self.cfg.train.alignment == 'None':
            return 0

class SimCLRLoss(nn.Module):

    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.CEL = torch.nn.CrossEntropyLoss()
        self.device = torch.device('cpu')
    
    def to(self, device):
        self.device = device
        self.CEL = self.CEL.to(device)
        return self

    def info_nce_loss(self, features):
        
        device = self.device

        # print(features.shape)
        bs = int(features.shape[0] // 2)
        labels = torch.cat([torch.arange(bs) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        # # Normlize the features according to subject
        # if stratified == 'stratified':
        #     features_str = features.clone()
        #     features_str[:bs, :] = (features[:bs, :] -  features[:bs, :].mean(
        #         dim=0)) / (features[:bs, :].std(dim=0) + 1e-3)
        #     features_str[bs:, :] = (features[bs:, :] -  features[bs:, :].mean(
        #         dim=0)) / (features[bs:, :].std(dim=0) + 1e-3)
        #     features = F.normalize(features_str, dim=1)
        # elif stratified == 'bn':
        #     features_str = features.clone()
        #     features_str = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-3)
        #     features = F.normalize(features_str, dim=1)
        # elif stratified == 'no':
        #     features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # Put the positive column at the end (when all the entries are the same, the top1 acc will be 0; while if the
        # positive column is at the start, the top1 acc might be exaggerated)
        logits = torch.cat([negatives, positives], dim=1)
        # The label means the last column contain the positive pairs
        labels = torch.ones(logits.shape[0], dtype=torch.long)*(logits.shape[1]-1)
        labels = labels.to(device)

        logits = logits / self.temperature
        return logits, labels

    def forward(self, features):
        # fea need to be normalized to 1
        self.to(features.device)
        logits, labels = self.info_nce_loss(features)
        loss = self.CEL(logits, labels)
        return loss, logits, labels




class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device
        return self

    def forward(self, features, labels=None, mask=None, modified=False):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].  need to be normalized to 1
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = self.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # features = F.normalize(features, dim=2)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob

        exp_logits = torch.exp(logits) * logits_mask

        if modified:
            nega_exp_logits_sum = (exp_logits*(~mask.bool())).sum(1)
            log_prob = torch.zeros_like(logits)
            for i in range(logits.shape[0]):
                for j in torch.nonzero(mask[i]).squeeze():
                    log_prob[i,j] = logits[i,j] - torch.log(nega_exp_logits_sum[i]+exp_logits[i,j])

        else:

            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss, exp_logits, mask