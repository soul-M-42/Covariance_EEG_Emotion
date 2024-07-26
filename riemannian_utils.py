import numpy as np
import torch

def frobenius_distance(matrix_a, matrix_b):
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Must have same shape")
    # 计算Frobenius距离
    if type(matrix_a) == torch.Tensor:
        distance = torch.linalg.norm(matrix_a - matrix_b, 'fro')
    if type(matrix_a) == np.ndarray:
        distance = np.linalg.norm(matrix_a - matrix_b, 'fro')
    return distance

def get_centroid(cov_matrices, tol=1e-5, max_iter=100):
    cov_matrices = np.array(cov_matrices)
    centroid = np.mean(cov_matrices, axis=0)
    
    for i in range(max_iter):
        prev_centroid = centroid.copy()
        weights = np.zeros(len(cov_matrices))
        for j, cov_matrix in enumerate(cov_matrices):
            weights[j] = 1.0 / frobenius_distance(cov_matrix, centroid)
        weights /= np.sum(weights)
        centroid = np.zeros_like(centroid)
        for j, cov_matrix in enumerate(cov_matrices):
            centroid += weights[j] * cov_matrix
        # 检查是否达到终止条件
        total_dis = 0
        for j, cov_matrix in enumerate(cov_matrices):
            total_dis += frobenius_distance(cov_matrix, centroid)
        delta = frobenius_distance(centroid, prev_centroid)
        # print(f'Ite {i} total_dis {total_dis / cov_matrices.shape[0]} delta {delta}')
        if frobenius_distance(centroid, prev_centroid) < tol:
            break
    return centroid