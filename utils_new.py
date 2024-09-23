import numpy as np
import torch

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
    # sequence: (B, time_steps, n_dims)
    B, n, n_dims = sequence.shape

    # Initialize output tensor
    sequence_new = torch.zeros_like(sequence)  # (B, n, n_dims)

    # Compute the mean over the time axis
    ave = torch.mean(sequence, dim=1)  # (B, n_dims)

    # Permute sequence to shape (B, n_dims, n)
    X = sequence.permute(0, 2, 1)  # (B, n_dims, n)

    # Initial state mean
    u0 = ave  # (B, n_dims)

    # Define constants
    V0 = 0.01
    A = 1.0
    T = 0.0001
    C = 1.0
    sigma = 1.0
    givenAll = 1

    # Initialize tensors
    P = torch.zeros((B, n_dims, n))
    u = torch.zeros((B, n_dims, n))
    V = torch.zeros((B, n_dims, n))
    K = torch.zeros((B, n_dims, n))

    # Initial Kalman gain
    K_init = V0 * C / (C * V0 * C + sigma)  # Scalar value
    K[:, :, 0] = K_init  # Broadcast to (B, n_dims)

    # Initial estimates
    u[:, :, 0] = u0 + K[:, :, 0] * (X[:, :, 0] - C * u0)
    V[:, :, 0] = (1 - K[:, :, 0] * C) * V0

    # Forward pass (Kalman Filter)
    for i in range(1, n):
        P[:, :, i - 1] = A * V[:, :, i - 1] * A + T
        Denominator = C * P[:, :, i - 1] * C + sigma
        K[:, :, i] = P[:, :, i - 1] * C / Denominator
        u[:, :, i] = A * u[:, :, i - 1] + K[:, :, i] * (X[:, :, i] - C * A * u[:, :, i - 1])
        V[:, :, i] = (1 - K[:, :, i] * C) * P[:, :, i - 1]

    # Backward pass (Rauch–Tung–Striebel smoother)
    if givenAll == 1:
        uAll = torch.zeros_like(u)
        VAll = torch.zeros_like(V)
        J = torch.zeros((B, n_dims, n))

        uAll[:, :, -1] = u[:, :, -1]
        VAll[:, :, -1] = V[:, :, -1]

        for ir in range(n - 1):
            i = n - 2 - ir
            J[:, :, i] = V[:, :, i] * A / P[:, :, i]
            uAll[:, :, i] = u[:, :, i] + J[:, :, i] * (uAll[:, :, i + 1] - A * u[:, :, i])
            VAll[:, :, i] = V[:, :, i] + J[:, :, i] * (VAll[:, :, i + 1] - P[:, :, i]) * J[:, :, i]

        X = uAll
    else:
        X = u

    # Permute back to original shape
    sequence_new = X.permute(0, 2, 1)  # (B, n, n_dims)

    return sequence_new