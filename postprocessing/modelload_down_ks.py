import torch
import numpy as np
import pennylane as qml
import torch.nn as nn
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Load and analyze KS QG trajectory models')
parser.add_argument('--n_qubits', type=int, default=10, help='Number of qubits used in the model (default: 10)')
parser.add_argument('--num_trajectories', type=int, default=500, help='Number of trajectories to analyze (default: 50)')
args = parser.parse_args()

# Load KS data - using the same preprocessing as in training
data_path = "../data/KS_data.npy"
data = np.load(data_path)  # shape: (1200, 2000, 512)
print(f"Original KS data shape: {data.shape}")

n_qubits = args.n_qubits
num_trajectories = args.num_trajectories

# Apply the same preprocessing as in training script
# Take first num_trajectories trajectories, time steps 200:456
selected_data = data[:num_trajectories, 200:456, :]  # shape: (num_trajectories, 256, 512)
print(f"Selected data shape before downsampling: {selected_data.shape}")

# Downsample spatial dimension: from 512 points to 256 points
downsample_factor = 2
downsampled_data = selected_data[:, :, ::downsample_factor]  # shape: (num_trajectories, 256, 256)
print(f"Data shape after spatial downsampling: {downsampled_data.shape}")

# Calculate global min and max values using the same data as training
global_min = downsampled_data.min()
global_max = downsampled_data.max()
print(f"Global value range after downsampling: {global_min:.6f} to {global_max:.6f}")

# Define histogram bins
num_bins = 2**n_qubits  # 1024 bins for 10 qubits
bin_edges = np.linspace(global_min, global_max, num_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Calculate the ground truth PDF using the same data preprocessing
print(f"Calculating ground truth PDF from {num_trajectories} trajectories...")
combined_hist = np.zeros(num_bins)
total_samples = 0

for traj in range(num_trajectories):
    trajectory_data = downsampled_data[traj]  # shape: (256, 256)
    flattened_data = trajectory_data.flatten()
    hist, _ = np.histogram(flattened_data, bins=bin_edges)
    if hist.sum() > 0:  # Avoid division by zero
        hist = hist / hist.sum()  # Normalize
        combined_hist += hist
        total_samples += 1

# Average the histograms
combined_hist = combined_hist / total_samples
print(f"Ground truth PDF calculated from {total_samples} trajectory samples")

# Define the same QG architecture as in KS training
try:
    dev = qml.device("lightning.qubit", wires=n_qubits)
except Exception as e:
    print("Using default.qubit instead of lightning.qubit")
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

# Load and generate distribution for a specific trajectory
def load_and_generate_trajectory(trajectory_index):
    model = QG()
    model_path = f"../models/model_ks_trajectories/model_trajectory_{trajectory_index}.pt"
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        with torch.no_grad():
            distribution = model().detach().cpu().numpy()
        return distribution
    except FileNotFoundError:
        print(f"Model for trajectory {trajectory_index} not found, skipping...")
        return None

# Load all trajectory models and collect their distributions
def load_all_trajectory_models():
    trajectory_distributions = []
    valid_models = 0
    
    for i in range(num_trajectories):
        distribution = load_and_generate_trajectory(i)
        if distribution is not None:
            trajectory_distributions.append(distribution)
            valid_models += 1
            print(f"Loaded QG model for trajectory {i}")
    
    print(f"Successfully loaded {valid_models} out of {num_trajectories} trajectory models")
    return np.array(trajectory_distributions) if trajectory_distributions else None

# Load all trajectory models
all_trajectory_distributions = load_all_trajectory_models()

# Plot the distributions
import matplotlib.pyplot as plt

if all_trajectory_distributions is not None:
    # Calculate global average distribution
    global_avg_distribution = np.mean(all_trajectory_distributions, axis=0)
    
    # Plot individual trajectory distributions
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Individual trajectory distributions
    plt.subplot(2, 2, 1)
    for i, distribution in enumerate(all_trajectory_distributions[:min(10, len(all_trajectory_distributions))]):
        # Map to physical values
        x_values = np.linspace(global_min, global_max, len(distribution))
        plt.plot(x_values, distribution, alpha=0.6, label=f"Trajectory {i}" if i < 5 else "")
    
    plt.title("Individual QG Trajectory Distributions")
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Global average distribution vs Ground truth
    plt.subplot(2, 2, 2)
    # Plot ground truth
    plt.plot(bin_centers, combined_hist, 'r-', linewidth=2, label="Ground Truth PDF")
    # Plot QG global average
    x_values = np.linspace(global_min, global_max, len(global_avg_distribution))
    plt.plot(x_values, global_avg_distribution, 'g-', linewidth=2, label="QG Global Average")
    
    plt.title("Ground Truth vs QG Global Average")
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Heatmap of all trajectory distributions
    plt.subplot(2, 2, 3)
    plt.imshow(all_trajectory_distributions, aspect='auto', cmap='viridis', 
               extent=[global_min, global_max, 0, len(all_trajectory_distributions)-1])
    plt.colorbar(label='Probability Density')
    plt.xlabel("Value")
    plt.ylabel("Trajectory Index")
    plt.title("QG Trajectory Distributions Heatmap")
    
    # Subplot 4: Distribution statistics
    plt.subplot(2, 2, 4)
    # Calculate mean and std across trajectories for each bin
    mean_dist = np.mean(all_trajectory_distributions, axis=0)
    std_dist = np.std(all_trajectory_distributions, axis=0)
    x_values = np.linspace(global_min, global_max, len(mean_dist))
    
    plt.plot(x_values, mean_dist, 'b-', linewidth=2, label="Mean")
    plt.fill_between(x_values, mean_dist - std_dist, mean_dist + std_dist, 
                     alpha=0.3, label="±1 Std")
    plt.plot(bin_centers, combined_hist, 'r--', linewidth=2, label="Ground Truth")
    
    plt.title("QG Statistics vs Ground Truth")
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"ks_trajectory_analysis_{n_qubits}qubits_{num_trajectories}traj.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and print some statistics
    print("\n=== Analysis Results ===")
    print(f"Number of qubits: {n_qubits}")
    print(f"Number of trajectories analyzed: {len(all_trajectory_distributions)}")
    print(f"Number of bins: {num_bins}")
    print(f"Value range: [{global_min:.6f}, {global_max:.6f}]")
    
    # Calculate KL divergence between ground truth and global average
    eps = 1e-10
    gt_normalized = combined_hist + eps
    gt_normalized = gt_normalized / gt_normalized.sum()
    QG_normalized = global_avg_distribution + eps
    QG_normalized = QG_normalized / QG_normalized.sum()
    
    kl_div = np.sum(gt_normalized * np.log(gt_normalized / QG_normalized))
    print(f"KL divergence (GT || QG): {kl_div:.6f}")
    
    # Save results
    np.save(f"trajectory_distributions_{n_qubits}qubits.npy", all_trajectory_distributions)
    np.save(f"regenerated_global_ks_pdf_10.npy", global_avg_distribution)
    np.save(f"ground_truth_pdf_{n_qubits}qubits.npy", combined_hist)
    
    print(f"\nResults saved:")
    print(f"- trajectory_distributions_{n_qubits}qubits.npy")
    print(f"- global_avg_distribution_{n_qubits}qubits.npy") 
    print(f"- ground_truth_pdf_{n_qubits}qubits.npy")
    print(f"- ks_trajectory_analysis_{n_qubits}qubits_{num_trajectories}traj.png")

else:
    print("No trajectory models found! Please make sure you have trained the models using QG_ks.py first.")
