import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def stratified_layerNorm(out, n_samples):
    n_samples = int(n_samples)
    n_subs = int(out.shape[0] / n_samples)
    out_str = out.clone()
    for i in range(n_subs):
        out_oneSub = out[n_samples*i: n_samples*(i+1)]
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0], -1, out_oneSub.shape[-1]).permute(0,2,1)
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0]*out_oneSub.shape[1], -1)
        # out_oneSub[torch.isinf(out_oneSub)] = -50
        # out_oneSub[torch.isnan(out_oneSub)] = -50
        out_oneSub_str = out_oneSub.clone()
        # We don't care about the channels with very small activations
        # out_oneSub_str[:, out_oneSub.abs().sum(dim=0) > 1e-4] = (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4] - out_oneSub[
        #     :, out_oneSub.abs().sum(dim=0) > 1e-4].mean(dim=0)) / (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4].std(dim=0) + 1e-3)
        out_oneSub_str = (out_oneSub - out_oneSub.mean(dim=0)) / (out_oneSub.std(dim=0) + 1e-3)
        out_str[n_samples*i: n_samples*(i+1)] = out_oneSub_str.reshape(n_samples, -1, out_oneSub_str.shape[1]).permute(0,2,1).reshape(n_samples, out.shape[1], out.shape[2], -1)
        # out_str[n_samples*i: n_samples*(i+1)] = out_oneSub_str
        # out_str[torch.isnan(out_str)]=1
    return out_str

def LDS(sequence):
    print(sequence.shape)
    # shape: (timeSample, n_dims)  timesample 为一个vid的采样数
    # print(sequence.shape) # (26, 256)   

    sequence_new = torch.zeros_like(sequence) # (26, 256)
    ave = torch.mean(sequence, axis=0)

    for te in range(sequence.shape[1]):
        X = sequence[:, te].reshape(1, -1) # (26,) to (1, 26)
        u0 = ave[te]

        V0 = 0.01
        A = 1
        T = 0.0001
        C = 1
        sigma = 1
        givenAll = 1

        [m, n] = X.shape # (1, 26)
        P = torch.zeros((m, m, n)) # (1, 1, 26)
        u = torch.zeros((m, n)) # (1, 26)
        V = torch.zeros((m, m, n)) # (1, 1, 26)
        K = torch.zeros((m, m, n)) # (1, 1, 26)

        K[:, :, 0] = V0*C / (C*V0*C + sigma)
        u[:, 0] = u0 + K[:, :, 0] * (X[:, 0] - C*u0)
        V[:, :, 0] = (torch.eye(m) - K[:, :, 0] * C) * V0

        for i in range(1, n):
            P[:, :, i-1] = A * V[:, :, i-1] * A + T
            K[:, :, i] = P[:, :, i-1] * C / (C * P[:, :, i-1] * C + sigma)
            u[:, i] = A * u[:, i-1] + K[:, :, i] * (X[:, i] - C*A*u[:, i-1])
            V[:, :, i] = (torch.eye(m) - K[:, :, i] * C) * P[:, :, i-1]

        if givenAll == 1:
            uAll = torch.zeros((m, n))
            VAll = torch.zeros((m, m, n))
            J = torch.zeros((m, m, n))
            uAll[:, n-1] = u[:, n-1]
            VAll[:, :, n-1] = V[:, :, n-1]

            for ir in range(n-1):
                i = n-2 - ir
                # print(i)
                J[:, :, i] = V[:, :, i] * A / P[:, :, i]
                uAll[:, i] = u[:, i] + J[:, :, i] * \
                    (uAll[:, i+1] - A * u[:, i])
                VAll[:, :, i] = V[:, :, i] + J[:, :, i] * \
                    (VAll[:, :, i+1] - P[:, :, i]) * J[:, :, i]

            X = uAll

        else:
            X = u

        sequence_new[:, te] = X
    return sequence_new

def LDS_new(sequence):
    # sequence: (B, n, n_dims)
    B, n, n_dims = sequence.shape

    # Compute the mean over the time axis
    ave = torch.mean(sequence, dim=1)  # (B, n_dims)

    # Permute sequence to shape (B, n_dims, n)
    X = sequence.permute(0, 2, 1)  # (B, n_dims, n)

    # Initial state mean
    u0 = ave  # (B, n_dims)

    # Define constants as tensors
    V0 = torch.tensor(0.01, dtype=sequence.dtype, device=sequence.device)
    A = torch.tensor(1.0, dtype=sequence.dtype, device=sequence.device)
    T = torch.tensor(0.0001, dtype=sequence.dtype, device=sequence.device)
    C = torch.tensor(1.0, dtype=sequence.dtype, device=sequence.device)
    sigma = torch.tensor(1.0, dtype=sequence.dtype, device=sequence.device)
    givenAll = 1

    # Initialize lists to collect values
    u_list = []
    V_list = []
    P_list = []
    K_list = []

    # Initial Kalman gain
    K_init = V0 * C / (C * V0 * C + sigma)  # Scalar tensor
    K0 = K_init.expand(B, n_dims)  # (B, n_dims)

    # Initial estimates
    u_prev = u0 + K0 * (X[:, :, 0] - C * u0)  # (B, n_dims)
    V_prev = (1 - K0 * C) * V0  # (B, n_dims)
    u_list.append(u_prev)
    V_list.append(V_prev)
    K_list.append(K0)

    # Forward pass (Kalman Filter)
    for i in range(1, n):
        P_prev = A * V_prev * A + T  # (B, n_dims)
        Denominator = C * P_prev * C + sigma  # (B, n_dims)
        K_curr = P_prev * C / Denominator  # (B, n_dims)
        u_curr = A * u_prev + K_curr * (X[:, :, i] - C * A * u_prev)  # (B, n_dims)
        V_curr = (1 - K_curr * C) * P_prev  # (B, n_dims)

        # Append current values to lists
        u_list.append(u_curr)
        V_list.append(V_curr)
        K_list.append(K_curr)
        P_list.append(P_prev)

        # Update previous values
        u_prev = u_curr
        V_prev = V_curr

    # Stack lists to form tensors of shape (B, n_dims, n)
    u = torch.stack(u_list, dim=2)  # (B, n_dims, n)
    V = torch.stack(V_list, dim=2)  # (B, n_dims, n)
    K = torch.stack(K_list, dim=2)  # (B, n_dims, n)

    # Backward pass (Rauch–Tung–Striebel smoother)
    if givenAll == 1:
        uAll_list = []
        VAll_list = []

        # Initialize with the last element
        uAll_prev = u[:, :, -1]  # (B, n_dims)
        VAll_prev = V[:, :, -1]  # (B, n_dims)

        # Insert at the beginning of the list
        uAll_list.insert(0, uAll_prev)
        VAll_list.insert(0, VAll_prev)

        for ir in range(n - 1):
            i = n - 2 - ir
            V_i = V[:, :, i]  # (B, n_dims)
            P_i = A * V_i * A + T  # (B, n_dims)

            J_i = V_i * A / P_i  # (B, n_dims)
            u_i = u[:, :, i]
            uAll_curr = u_i + J_i * (uAll_prev - A * u_i)
            VAll_curr = V_i + J_i * (VAll_prev - P_i) * J_i

            # Insert at the beginning of the list
            uAll_list.insert(0, uAll_curr)
            VAll_list.insert(0, VAll_curr)

            # Update previous values
            uAll_prev = uAll_curr
            VAll_prev = VAll_curr

        # Stack lists to form tensors
        uAll = torch.stack(uAll_list, dim=2)  # (B, n_dims, n)

        X = uAll
    else:
        X = u

    # Permute back to original shape
    sequence_new = X.permute(0, 2, 1)  # (B, n, n_dims)
    return sequence_new

def save_img(data, filename='image_with_colorbar.png', cmap='viridis'):
    # return
    print('called')
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    fig, ax = plt.subplots()
    cax = ax.imshow(data, cmap=cmap)
    fig.colorbar(cax)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def save_batch_images(data, folder='heatmaps', cmap='viridis'):
    # 创建保存热力图的文件夹
    if not os.path.exists(folder):
        os.makedirs(folder)

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # 假设数据的形状为 [B, W, D]
    B, W, D = data.shape

    # 循环遍历每个批次，生成热力图
    for i in range(B):
        filename = os.path.join(folder, f'heatmap_{i}.png')
        fig, ax = plt.subplots()
        cax = ax.imshow(data[i], cmap=cmap)
        fig.colorbar(cax)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)