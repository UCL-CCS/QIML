import torch
import numpy as np
import pennylane as qml
import torch.nn as nn

data_path = "../data/TCF_data.npy"
gt_data = np.load(data_path)

# Calculate the global min and max values for the first 10 images
all_speeds = []
num_images = 595  # Using only the first 10 images as requested
for i in range(num_images): 
    img = gt_data[i, :, :, 0]  # Get x component
    all_speeds.append(img.flatten())
all_speeds = np.concatenate(all_speeds)
global_min, global_max = all_speeds.min(), all_speeds.max()
print(f"Global speed range for first 10 images: {global_min:.6f} to {global_max:.6f}")

# Define histogram bins
num_bins = 128  # You can adjust this for desired resolution
bin_edges = np.linspace(global_min, global_max, num_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # For plotting

# Calculate the ground truth PDF
hist, _ = np.histogram(all_speeds, bins=bin_edges)

# Define histogram bins
num_bins = 128    # You can adjust this for desired resolution
# Calculate the combined PDF from the first 10 images
combined_hist = np.zeros(num_bins)
for i in range(num_images):
    img = gt_data[i, :, :, 0]
    hist, _ = np.histogram(img, bins=bin_edges)
    if hist.sum() > 0:  # Avoid division by zero
        hist = hist / hist.sum()  # Normalize
        combined_hist += hist

# Average the histograms
combined_hist = combined_hist / num_images

# Plot the ground truth PDF
import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.plot(bin_centers, combined_hist, 'r-', linewidth=2)
# plt.title("Ground Truth PDF")
# plt.xlabel("Speed (v)")
# plt.ylabel("Probability")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("ground_truth_pdf.png")
# plt.show()


# Define the same QG architecture as before
n_qubits = 15

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

# Load and generate distribution for a specific model
def load_and_generate(model_index):
    model = QG()
    model_path = f"../modeldown/model_{model_index}.pt"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        distribution = model().detach().cpu().numpy()
    return distribution
    # return distribution / distribution.sum()  # Normalize

# Example usage:
# Load specific model
model_index = 0  # Change this to load different models
distribution = load_and_generate(model_index)

# If you want to load all models and average their distributions
def load_all_models(num_images=595):
    global_pdf_accum = np.zeros(32768)  # 2^16 = 65536 states
    for i in range(num_images):
        try:
            distribution = load_and_generate(i)
            global_pdf_accum += distribution
            print(f"Loaded model {i}")
        except FileNotFoundError:
            print(f"Model {i} not found, skipping...")
    return global_pdf_accum / num_images

# Generate and save the global distribution
# Load all models and plot individual and global distributions
global_pdf = load_all_models()

# Plot the distributions
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Plot individual model distributions
num_images = 595    # Same as in load_all_models
# Store all resampled distributions for calculating the global average
all_resampled_distributions = []
global_bin_centers = None

# First, determine the global min and max across all models to ensure consistent range
all_values_min = float('inf')
all_values_max = float('-inf')
all_distributions = []

# First pass to find global min and max
for i in range(num_images):
    try:
        distribution = load_and_generate(i)
        normalized_distribution = distribution / np.sum(distribution)
        all_distributions.append(normalized_distribution)
        all_values_min = min(all_values_min, distribution.min())
        all_values_max = max(all_values_max, distribution.max())
    except FileNotFoundError:
        continue

# Create a common set of bin centers for all distributions
global_bin_centers = np.linspace(all_values_min, all_values_max, 128)
bin_width = (all_values_max - all_values_min) / (len(global_bin_centers) - 1)
all_resampled_distributions = []

# Second pass to resample all distributions to the common bin centers
for i, distribution in enumerate(all_distributions):
    bin_probs = np.zeros_like(global_bin_centers)
    for j, prob in enumerate(distribution):
        # Map the original index to a value in our global range
        orig_value = all_values_min + (j / (len(distribution) - 1)) * (all_values_max - all_values_min)
        
        # Find the closest bin center
        bin_idx = int(np.round((orig_value - all_values_min) / bin_width))
        if 0 <= bin_idx < len(global_bin_centers):
            bin_probs[bin_idx] += prob
    
    # Ensure the resampled distribution is normalized
    bin_probs = bin_probs / np.sum(bin_probs) if np.sum(bin_probs) > 0 else bin_probs
    all_resampled_distributions.append(bin_probs)
    plt.plot(global_bin_centers, bin_probs, alpha=0.3, label=f"Model {i}")

# Calculate the average probability distribution (global probability)
if all_resampled_distributions:
    global_resampled_pdf = np.mean(all_resampled_distributions, axis=0)
    plt.plot(global_bin_centers, global_resampled_pdf, 'k-', linewidth=2, label="Global Average PDF")
    # Save this global resampled PDF
    #np.save("global_resampled_pdf.npy", global_resampled_pdf)
    #print("Global resampled PDF has been saved to global_resampled_pdf.npy")

# Plot the global distribution with thicker line
# plt.plot(range(32768), global_pdf[:], 'k-', linewidth=2, label="Global PDF")

plt.title("Quantum Circuit Born Machine Distributions")
plt.xlabel("State Index")
plt.ylabel("Probability")
#plt.legend()
plt.tight_layout()
#plt.savefig("QG_distributions.png")
#plt.show()
np.save("regenerated_global_speed_pdf_2.npy", global_pdf)
print("Regenerated global speed distribution has been saved")

# After you've calculated both distributions, create a comparison plot
plt.figure(figsize=(12, 6))

# Plot the ground truth PDF
plt.plot(bin_centers, combined_hist, 'r-', linewidth=2, label="Ground Truth PDF")

# Calculate the average QG distribution
if all_resampled_distributions:
    global_resampled_pdf = np.mean(all_resampled_distributions, axis=0)
    
    # You may need to rescale the x-axis of the QG distribution to match the ground truth
    # This assumes you want to map the QG distribution to the same range as the ground truth
    QG_x = np.linspace(global_min, global_max, len(global_resampled_pdf))
    
    # Plot the QG global average
    plt.plot(QG_x, global_resampled_pdf, 'g-', linewidth=2, label="QG Global Average")
    
    # # Optionally, plot individual model distributions with low opacity
    # for i, dist in enumerate(all_resampled_distributions):
    #     if i < 10:  # Limit to first 10 models to avoid cluttering
    #         plt.plot(QG_x, dist, 'b-', alpha=0.1)

plt.title("Comparison of Ground Truth and QG Distributions")
plt.xlabel("Speed (v)")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("distribution_comparison.png")
#plt.show()
