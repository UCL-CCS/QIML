import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append('../')
from utilities import *

sys.path.append('../lib')
from pfnn_consist_2d import *

from timeit import default_timer
import scipy.io
import os
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import matplotlib.pyplot as plt

# Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")


torch.manual_seed(0)
np.random.seed(0)

data = np.load('../data/train_set_vxyz_s2_1_64.npy')[..., 0]
#data = torch.tensor(data).unsqueeze(0)
print('Data shape:', data.shape)

# Main
processing = True
save = False
load_model = False

batch_size = 32
epochs = 50
learning_rate = 0.0001
scheduler_step = 10
scheduler_gamma = 0.5
gamma_2 = 1

sub = 3  # spatial subsample
S = 192  # size of image, also the domain size
s = S // sub
in_dim = 1
out_dim = 1
T_in = 2
T = 140
T_out = T_in + T

steps = 1
steps_back = 1
backward = True

if processing:
    t1 = default_timer()
    data = torch.tensor(data, dtype=torch.float32)[..., ::sub, ::sub]

    episode_samples = 42
    test_samples = 20
    data_sampled_train = data[torch.randperm(data.size(0))[:episode_samples], ...]
    data_sampled_test = data[torch.randperm(data.size(0))[:test_samples], ...]

    train_a = data_sampled_train[:, T_in - 1:T_out - 1].reshape(-1, s, s)
    train_u = data_sampled_train[:, T_in:T_out].reshape(-1, s, s)
    test_a = data_sampled_test[:, T_in - 1:T_out - 1].reshape(-1, s, s)
    test_u = data_sampled_test[:, T_in:T_out].reshape(-1, s, s)

    if save:
        scipy.io.savemat('./data/2D_NS_Re500_train.mat', mdict={'a': train_a.numpy(), 'u': train_u.numpy()})
        scipy.io.savemat('./data/2D_NS_Re500_test.mat', mdict={'a': test_a.numpy(), 'u': test_u.numpy()})

    t2 = default_timer()

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size,
                                          shuffle=False)

if not load_model:
    model = KoopmanAE_2d_trans(in_dim, out_dim, dim=16, num_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8],
                               ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', steps=1,
                               steps_back=1).to(device)
else:
    path_model = '../mainmodel/model.pt'
    model = torch.load(path_model).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

trainloss = LpLoss(size_average=False)
id_loss = LpLoss(size_average=False)
testloss = LpLoss(size_average=False)

train_l2_list = []
train_id_list = []
train_l2_back_list = []
train_consist_list = []
test_loss_list = []



for ep in range(1, epochs + 1):
    model.train()
    train_l2, train_id, train_l2_back, train_consist = [], [], [], []

    for x, y in tqdm.tqdm(train_loader):
        bs = x.shape[0]
        x, y = x.to(device).view(bs, s, s, 1), y.to(device).view(bs, s, s, 1)

        loss_fwd, loss_identity = 0, 0
        for k in range(1):
            out_pred, out_identity = model(x, mode='forward')
            loss_fwd += trainloss(out_pred[0].permute(0, 2, 3, 1), y)
            loss_identity += id_loss(out_identity[0].permute(0, 2, 3, 1), x)

        if steps_back:
            loss_bwd = 0
            for k in range(1):
                out_back_pred, out_back_identity = model(y, mode='backward')
                loss_bwd += trainloss(out_back_pred[0].permute(0, 2, 3, 1), x)

            predicted_velocity = out_pred[0]
            if predicted_velocity.shape[1] > 1:
                velocity_magnitude = torch.sqrt(torch.sum(predicted_velocity ** 2, dim=1))
            else:
                velocity_magnitude = torch.abs(predicted_velocity.squeeze(1))

            min_speed, max_speed = 0, 10
            bin_edges = torch.linspace(min_speed, max_speed, num_bins + 1, device=device)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
            velocity_magnitude = torch.clamp(velocity_magnitude, min_speed, max_speed)


            hist = torch.histc(velocity_magnitude, bins=num_bins, min=bin_edges.min().item(),
                               max=bin_edges.max().item())
            epsilon = 1e-8

            predicted_distribution = (hist + epsilon) / (torch.sum(hist) + num_bins * epsilon)

            if ep % 5 == 0:
                print(f"Epoch {ep} | Predicted distribution sum: {predicted_distribution.sum().item():.6f}")

            if torch.sum(hist) == 0:
                print("Warning: Histogram sum is 0! Check velocity_magnitude range.")


            loss = loss_fwd + loss_identity + 0.1 * loss_bwd
        else:
            loss = loss_fwd + loss_identity

        train_l2.append(loss_fwd.item())
        train_id.append(loss_identity.item())
        train_l2_back.append(loss_bwd.item())
        train_consist.append(mmd_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2 = []
    with torch.no_grad():
        for x, y in test_loader:
            bs = x.shape[0]
            x, y = x.to(device).view(bs, s, s, 1), y.to(device).view(bs, s, s, 1)
            out = model(x)[0]
            test_l2.append(testloss(out[0].reshape(bs, s, s, 1), y).item())

    test_loss_list.append(np.mean(test_l2))

    scheduler.step()
    print(
        f"Epoch {ep} | Train Loss: {np.mean(train_l2):.3f}, mmd: {np.mean(train_consist):.3f}, Test Loss: {np.mean(test_l2):.3f}")

torch.save(model, '../mainmodel/final_model.pt')
print('Training complete. Model saved.')
