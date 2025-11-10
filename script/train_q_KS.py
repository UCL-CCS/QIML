import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import matplotlib.pyplot as plt
from tqdm import tqdm

from timeit import default_timer
import scipy.io
import os

import sys
sys.path.append('../')
sys.path.append('../lib')
from vae_base import *
from utilities import *

torch.manual_seed(0)
np.random.seed(0)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")


QG_prior_np = np.load(".../postprocessing/Q-Prior_ks_pdf_10.npy")
print(f"QG prior shape: {QG_prior_np.shape}")


QG_prior_np /= QG_prior_np.sum()
QG_prior = torch.tensor(QG_prior_np, dtype=torch.float32, device=device)
print(f"QG prior dimensions: {len(QG_prior)}")

# Data loading and splitting
data_raw = np.load('../data/KS_data.npy')
data_tensor = torch.tensor(data_raw, dtype=torch.float)
print(data_tensor.shape)

# Data parameters
n_trajectories = data_tensor.shape[0]  # 1200 trajectories
n_train = int(0.8 * n_trajectories)    # 80% for training
n_val = int(0.1 * n_trajectories)      # 10% for validation
n_test = n_trajectories - n_train - n_val  # remaining 10% for testing

# Shuffle indices
indices = torch.randperm(n_trajectories)
train_indices = indices[:n_train]
val_indices = indices[n_train:n_train+n_val]
test_indices = indices[n_train+n_val:]

# Split data
data_train = data_tensor[train_indices]
data_val = data_tensor[val_indices]
data_test = data_tensor[test_indices]

# Parameters for time series
T_in = 200  # skip first 100 timesteps
T = 1600    # seconds to extract from each trajectory
T_out = T_in + T
step = 1    # Seconds to learn solution operator
batch_size = 400
steps_per_sec = 10  # given temporal subsampling

# Sample episodes for training
episode_samples = 900
data_sampled_train = data_train[torch.randperm(data_train.size(0))[:episode_samples],:,:]
data_sampled_val = data_val  # use all validation data
data_sampled_test = data_test  # use all test data

# Create input-output pairs
train_a = data_sampled_train[:,T_in-1:T_out-1,:].reshape(-1, 512)  # current states
train_u = data_sampled_train[:,T_in:T_out,:].reshape(-1, 512)      # next states
val_a = data_sampled_val[:,T_in-1:T_out-1,:].reshape(-1, 512)
val_u = data_sampled_val[:,T_in:T_out,:].reshape(-1, 512)
test_a = data_sampled_test[:,T_in-1:T_out-1,:].reshape(-1, 512)
test_u = data_sampled_test[:,T_in:T_out,:].reshape(-1, 512)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_a, train_u), 
    batch_size=batch_size, 
    shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(val_a, val_u), 
    batch_size=batch_size, 
    shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_a, test_u), 
    batch_size=batch_size, 
    shuffle=False
)

# Update dimensions for the model
in_dim = 512   # KS data has 512 dimensions
out_dim = 512

# experiment parameters
epochs = 100 # 1000
learning_rate = 0.0001# 0.0005
scheduler_step = 10
scheduler_gamma = 0.5

gamma_1 = 1
gamma_2 = 1000  # MMD loss weight
gamma_3 = 1000  # KL loss weight
gamma_4 = 15 # peak region loss weight
latent_dim = 4 # origin 10

nonlinearity = nn.ReLU
# model parameters
encoder_layers = [in_dim, in_dim*10, in_dim*10, in_dim*latent_dim] # structure [input_x_{t}, hidden1in, hidden1out, [mu_1, var_1]]
forward_layers = [in_dim*latent_dim, in_dim*latent_dim] # structure [latent_in, latent_out, [mu_2, var_2]], forward_matrix = dense(latent_in, latent_out).weight()
decoder_layers = [in_dim*latent_dim, in_dim*10, in_dim*10, out_dim] # structure [z_{t+1}, hidden1in, hidden1out, output_x_{t+1}]

model = KoopmanAE(
    encoder_layers=encoder_layers,
    decoder_layers=decoder_layers,
    steps=1,
    steps_back=1,
    init_scale=1,
    nonlinearity=nonlinearity
).to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
# loss function, MSE loss & KL divergence
pred_loss = nn.MSELoss(reduction='sum').to(device)
id_loss = nn.MSELoss(reduction='sum').to(device)
testloss = nn.MSELoss(reduction='sum').to(device)
testloss_1sec = nn.MSELoss(reduction='sum').to(device)

relu = torch.nn.ReLU().to(device)


def peak_region_loss(pred_dist, target_dist, threshold=0.02):

    peak_mask = (target_dist > threshold)

    peak_loss = F.mse_loss(pred_dist[peak_mask], target_dist[peak_mask])
    return peak_loss


def compute_kl_divergence(predicted_distribution, prior_distribution):

    epsilon = 1e-8
    predicted_distribution = predicted_distribution + epsilon
    predicted_distribution /= predicted_distribution.sum()

    prior_distribution = prior_distribution + epsilon
    prior_distribution /= prior_distribution.sum()

    return F.kl_div(predicted_distribution.log(), prior_distribution, reduction='batchmean')

def contractive_loss(forward_weights):
      forward_weights = forward_weights.cpu()
      # forward_weights * forward_weights.T, by Frobenius norm for egienvalues, root squared omitted
      forward_square = torch.matmul(forward_weights, forward_weights.T)
      forward_eigen_CPU = torch.linalg.eigvals(forward_square).real
      forward_eigen = forward_eigen_CPU.to(device)
      loss = relu(forward_eigen-1)
      return loss.sum(), forward_eigen_CPU

train_l2_list = []
train_kl_list = []
train_mmd_list = []
test_l2_list = []

for ep in range(1, epochs + 1):
    model.train()
    t1 = default_timer()
    one_sec_count = 0
    train_l2 = []
    train_id = []
    train_kl = []
    train_mmd = []
    train_contractive = []

    for x, y in tqdm(train_loader):
        x = x.to(device).view(-1, out_dim)
        y = y.to(device).view(-1, out_dim)

        out_pred, out_identity = model(x, mode='forward')
        
        out_pred = out_pred[0]
        out_identity = out_identity[0]

        loss_pred = pred_loss(out_pred, y)
        loss_id = id_loss(out_identity, x)
        
        rollout_steps = 10 
        rollout_distributions = []
        temperature = 1.0  
        
        target_bins = 128  

        current_input = x.view(-1, out_dim)
        for step in range(rollout_steps):
            out_step, _ = model(current_input, mode='forward')
            trajectory_step = out_step[0]  # shape: [batch_size, 512]
            

            trajectory_magnitude_step = torch.abs(trajectory_step)  # [batch_size, 512]
            
            distribution_step = torch.softmax(trajectory_magnitude_step.flatten() / temperature, dim=0)

            current_size = len(distribution_step)
            if current_size >= target_bins:
                group_size = current_size // target_bins
                distribution_step = distribution_step[:target_bins * group_size].view(target_bins, group_size).mean(dim=1)
            else:
                distribution_step = F.interpolate(
                    distribution_step.unsqueeze(0).unsqueeze(0), 
                    size=target_bins, 
                    mode='linear', 
                    align_corners=False
                ).squeeze()
            
            epsilon = 1e-8
            distribution_step = (distribution_step + epsilon) / (distribution_step.sum() + target_bins * epsilon)
            
            rollout_distributions.append(distribution_step)
            current_input = out_step[0]  

        avg_distribution = torch.stack(rollout_distributions).mean(dim=0)  # [target_bins]
        
        avg_distribution = avg_distribution / avg_distribution.sum()
        
        QG_size = len(QG_prior)
        if QG_size >= target_bins:
            QG_group_size = QG_size // target_bins
            QG_binned = QG_prior[:target_bins * QG_group_size].view(target_bins, QG_group_size).mean(dim=1)
        else:
            QG_binned = F.interpolate(
                QG_prior.unsqueeze(0).unsqueeze(0), 
                size=target_bins, 
                mode='linear', 
                align_corners=False
            ).squeeze()
        
        QG_binned_normalized = QG_binned / QG_binned.sum()
        
        loss_KL = F.kl_div(avg_distribution.log(), QG_binned_normalized, reduction='batchmean')
        loss_MMD = torch.norm(avg_distribution - QG_binned_normalized, p=2)
        
        if ep == 1 and len(train_l2) == 0: 
            print(f"KS trajectory shape: {trajectory_step.shape}")
            print(f"Trajectory magnitude shape: {trajectory_magnitude_step.shape}")
            print(f"Distribution before binning: {len(trajectory_magnitude_step.flatten())}")
            print(f"Target bins: {target_bins}")
            print(f"QG original size: {QG_size} -> binned: {len(QG_binned)}")
            print(f"avg_distribution shape: {avg_distribution.shape}")
            print(f"avg_distribution requires_grad: {avg_distribution.requires_grad}")
            print(f"loss_KL requires_grad: {loss_KL.requires_grad}")
            print(f"loss_MMD requires_grad: {loss_MMD.requires_grad}")

        # loss_contractive, forward_eigen = contractive_loss(model.dynamics.dynamics.weight)
        loss = loss_pred + loss_id + gamma_2 * loss_MMD + gamma_3 * loss_KL #+ gamma_4 * peak_loss
        
        train_l2.append(loss_pred.item())
        train_id.append(loss_id.item())
        train_kl.append(loss_KL.item())
        train_mmd.append(loss_MMD.item())
        # train_contractive.append(loss_contractive.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_l2_list.append(np.mean(train_l2))
    train_kl_list.append(np.mean(train_kl))
    train_mmd_list.append(np.mean(train_mmd))
    # train_contract_list.append(np.mean(train_contractive))

    # model.eval()
    test_l2 = []
    test_l2_1_sec = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device).view(-1, out_dim)
            y = y.to(device).view(-1, out_dim)

            out_pred, out_identity = model(x, mode='forward')
            out = out_pred[0]
            out = out.reshape(-1, out_dim)
            test_l2.append(testloss(out, y).item())

            x_subsample = x[::steps_per_sec]
            x_1sec = x_subsample[:-2] # inputs
            y_1sec = x_subsample[1:-1] # ground truth
            out_1sec = x_1sec
            for i in range(10):
                out_1sec_pred, _ = model(out_1sec, mode='forward')
                out_1sec = out_1sec_pred[0]
            test_1_sec_loss = testloss_1sec(out_1sec.reshape(-1, out_dim), y_1sec)
            test_l2_1_sec.append(test_1_sec_loss.item())
            one_sec_count += (int)(y_1sec.shape[0]) 
    test_l2_list.append(np.mean(test_l2))
    t2 = default_timer()
    scheduler.step()
    print("Epoch " + str(ep) + " Train L2 err:", "{0:.{1}f}".format(train_l2_list[-1], 3), 
    "Train ID err:", "{0:.{1}f}".format(np.mean(train_id), 3), 
    "Train KL err:", "{0:.{1}f}".format(train_kl_list[-1], 3),
    "Train MMD err:", "{0:.{1}f}".format(train_mmd_list[-1], 3),
    # "Train Contractive err:", "{0:.{1}f}".format(train_contract_list[-1], 3),
    "Test L2 err:", "{0:.{1}f}".format(test_l2_list[-1], 3), "Test L2 err over 1 sec:", "{0:.{1}f}".format(np.sum(test_l2_1_sec)/(one_sec_count), 3))

    # Save model based on best test L2 loss
    if ep == 1:
        best_test_loss = test_l2_list[-1]
        path = 'KS_model_with_QG_128dim'
        path_model = '../model/'+path
        os.makedirs(path_model, exist_ok=True)
        path_model = path_model + '/model_with_q&id.pt'
        torch.save(model, path_model)
        print('Model saved at', path_model, 'with test L2 loss:', "{0:.{1}f}".format(best_test_loss, 3))
    elif test_l2_list[-1] < best_test_loss:
        best_test_loss = test_l2_list[-1]
        path = 'KS_model_with_QG_128dim'
        path_model = '../model/'+path
        os.makedirs(path_model, exist_ok=True)
        path_model = path_model + '/model_with_q&id.pt'
        torch.save(model, path_model)
        print('Model saved at', path_model, 'with improved test L2 loss:', "{0:.{1}f}".format(best_test_loss, 3))
