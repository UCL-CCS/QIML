import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import time
import os
import argparse

# -------------------------------
# 0. Parse command line arguments
# -------------------------------
parser = argparse.ArgumentParser(description='QG training for KS equation time evolution')
parser.add_argument('--n_qubits', type=int, default=10, help='Number of qubits to use (default: 10)')
parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs (default: 500)')
parser.add_argument('--num_trajectories', type=int, default=500, help='Number of trajectories to train (default: 50)')
args = parser.parse_args()

# -------------------------------

n_qubits = args.n_qubits
num_trajectories = args.num_trajectories

data_path = "../data/KS_data.npy"
data = np.load(data_path)  # shape: (1200, 2000, 512)
print(f"Original data shape: {data.shape}")

selected_data = data[:num_trajectories, 200:456, :]  # shape: (50, 256, 512)
print(f"Selected data shape before downsampling: {selected_data.shape}")

downsample_factor = 4
downsampled_data = selected_data[:, :, ::downsample_factor]  # shape: (num_trajectories, 256, 128)
print(f"Data shape after spatial downsampling: {downsampled_data.shape}")

global_min = downsampled_data.min()
global_max = downsampled_data.max()
print(f"Global value range after downsampling: {global_min} to {global_max}")

num_bins = 2**n_qubits  # 1024 bins for 10 qubits
bin_edges = np.linspace(global_min, global_max, num_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# -------------------------------

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def QG_circuit(params):
    num_layers = 8
    for l in range(num_layers):
        for i in range(n_qubits):
            qml.RY(params[l * 3 * n_qubits + i], wires=i)
            qml.RZ(params[l * 3 * n_qubits + i + n_qubits], wires=i)
            qml.RX(params[l * 3 * n_qubits + i + 2 * n_qubits], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
    return qml.probs(wires=range(n_qubits))

class QG(nn.Module):
    def __init__(self):
        super(QG, self).__init__()
        num_layers = 8
        self.params = nn.Parameter(torch.randn(num_layers * 3 * n_qubits))
    
    def forward(self):
        return QG_circuit(self.params)

# -------------------------------

def mmd_loss_direct(probs, target_samples, bin_centers, sigma=1.0):

    if not isinstance(target_samples, torch.Tensor):
        target_samples = torch.tensor(target_samples, dtype=torch.float32)
    if not isinstance(bin_centers, torch.Tensor):
        bin_centers = torch.tensor(bin_centers, dtype=torch.float32)
    

    device = probs.device
    target_samples = target_samples.to(device)
    bin_centers = bin_centers.to(device)
    

    bin_indices = torch.searchsorted(bin_centers, target_samples, right=True) - 1
    bin_indices = torch.clamp(bin_indices, 0, len(bin_centers) - 1)
    

    target_one_hot = torch.zeros(len(bin_centers), device=device)
    for idx in bin_indices:
        target_one_hot[idx] += 1
    target_one_hot = target_one_hot / target_one_hot.sum()

    mmd_loss = torch.sum((probs - target_one_hot) ** 2)
    
    return mmd_loss

# -------------------------------

def train_QG_for_trajectory(trajectory_data, bin_centers, num_epochs=200, lr=0.002, trajectory_index=0):
    """
    trajectory_data: 2D numpy array, shape (256, 128) representing the spatio-temporal data for one trajectory
    """
    device_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device_torch}")
    

    flattened_data = trajectory_data.flatten()
    num_samples = min(5000, len(flattened_data)) 
    target_samples = np.random.choice(flattened_data, size=num_samples, replace=False)

    target_samples = torch.tensor(target_samples, dtype=torch.float32, device=device_torch)
    bin_centers_tensor = torch.tensor(bin_centers, dtype=torch.float32, device=device_torch)
    
    model = QG().to(device_torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    
    print(f"Training trajectory {trajectory_index}...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        

        probs = model()
        

        loss = mmd_loss_direct(probs, target_samples, bin_centers_tensor, sigma=1.0)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Trajectory {trajectory_index}, Epoch {epoch}: MMD Loss = {loss.item()}")
    
    learned_pdf = model().detach().cpu().numpy()
    

    model_dir = f"../models/model_ks_trajectories_128dim/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f"model_trajectory_{trajectory_index}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model for trajectory {trajectory_index} saved to {model_path}")
    
    return learned_pdf

# -------------------------------

print(f"Processing {num_trajectories} trajectories...")
print(f"Each trajectory has shape: {downsampled_data.shape[1:]} (time_steps, spatial_points)")
print(f"Each trajectory will have its own QG model with {n_qubits} qubits")
print(f"Models will be saved in ../models/model_ks_trajectories_128dim/ directory")

trajectory_pdfs = []
total_start_time = time.time()  

for traj_idx in range(num_trajectories):
    start_time = time.time()
    print(f"\nProcessing trajectory {traj_idx + 1}/{num_trajectories}...")
    

    trajectory_data = downsampled_data[traj_idx]  # shape: (256, 128)
    

    learned_pdf = train_QG_for_trajectory(trajectory_data, bin_centers, num_epochs=args.epochs, lr=0.002, trajectory_index=traj_idx)
    trajectory_pdfs.append(learned_pdf)
    
    end_time = time.time()
    step_time = end_time - start_time
    total_time = end_time - total_start_time
    print(f"Time for trajectory {traj_idx}: {step_time:.2f} seconds")
    print(f"Total training time so far: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


total_time = time.time() - total_start_time
print(f"\nTraining completed!")
print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


trajectory_pdfs = np.array(trajectory_pdfs)
np.save("ks_Q-Prior.npy", trajectory_pdfs)
print(f"Trajectory PDFs saved to ks_Q-Prior.npy with shape: {trajectory_pdfs.shape}")

# -------------------------------

plt.figure(figsize=(14, 8))
plt.imshow(trajectory_pdfs, aspect='auto', cmap='viridis', 
           extent=[bin_centers[0], bin_centers[-1], 0, num_trajectories-1])
plt.colorbar(label='Probability Density')
plt.xlabel("Value Bins")
plt.ylabel("Trajectory Index")
plt.title(f"QG Learned Q-Prior for {num_trajectories} KS Trajectories (MMD)")
plt.savefig("ks_Q-Prior_mmd.png", dpi=300, bbox_inches='tight')
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i in range(min(6, num_trajectories)):
    ax = axes[i]
    im = ax.imshow(downsampled_data[i].T, aspect='auto', cmap='viridis')
    ax.set_title(f'Trajectory {i}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Spatial Point')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("ks_Q-Prior.png", dpi=300, bbox_inches='tight')
plt.show()
