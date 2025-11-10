import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import time
import os
import argparse

from skimage.transform import resize

# -------------------------------
# 0. Parse command line arguments
# -------------------------------
parser = argparse.ArgumentParser(description='QG training with configurable number of qubits')
parser.add_argument('--n_qubits', type=int, default=15, help='Number of qubits to use (default: 15)')
parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs (default: 500)')
args = parser.parse_args()


n_qubits = args.n_qubits  # Use the command line argument

data_path = "../data/kf_2d_re1000_256_120seed.npy"
data = np.load(data_path)  # shape: (120, 320, 256, 256)
data = data[0]  # Only use the first trajectory, shape: (320, 256, 256)

num_images = data.shape[0]  # 595 张图
H, W = data.shape[1], data.shape[2]  # 192 x 192


all_speeds = []
for i in range(num_images):
    img = data[i, :, :]  # shape: (192,192)
    all_speeds.append(img)
all_speeds = np.concatenate([img.flatten() for img in all_speeds])
global_min = all_speeds.min()
global_max = all_speeds.max()
print("Global speed range：", global_min, global_max)


num_bins = 2**n_qubits
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

def train_QG_on_image(target_samples, bin_centers, num_epochs=200, lr=0.002, global_index=0):

    device_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device_torch}")
    
    target_samples = torch.tensor(target_samples, dtype=torch.float32, device=device_torch)
    bin_centers_tensor = torch.tensor(bin_centers, dtype=torch.float32, device=device_torch)
    
    model = QG().to(device_torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        probs = model()
        
        loss = mmd_loss_direct(probs, target_samples, bin_centers_tensor, sigma=1.0)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: MMD Loss = {loss.item()}")
    
    learned_pdf = model().detach().cpu().numpy()

    model_dir = f"../models/model_kf_{n_qubits}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f"model_{global_index}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    learned_pdf = learned_pdf / learned_pdf.sum()
    return learned_pdf

target_res_one = (181, 181)  # 181x181 = 32761
pad_length = 32768 - (target_res_one[0] * target_res_one[1])  # 32768 - 32761 = 7

global_pdf_accum = np.zeros(num_bins)  
print(f"Total images: {num_images}")

for i in range(num_images):
    start_time = time.time()
    print(f"Processing image {i + 1}/{num_images} ...")

    img = data[i, :, :]
    
    img_resized = resize(img, target_res_one, anti_aliasing=True)

    vec = img_resized.flatten()
    if pad_length > 0:
        vec = np.pad(vec, (0, pad_length), mode='constant')


    img_flat = vec.flatten()
    num_samples = min(5000, len(img_flat))  
    target_samples = np.random.choice(img_flat, size=num_samples, replace=False)

    learned_pdf = train_QG_on_image(target_samples, bin_centers, num_epochs=500, lr=0.002, global_index=i)
    global_pdf_accum += learned_pdf

    end_time = time.time()
    print(f"Time for image {i + 1}: {end_time - start_time:.2f} seconds")

global_pdf = global_pdf_accum / num_images

# -------------------------------

plt.figure(figsize=(10, 5))
plt.plot(bin_centers, global_pdf, label="Q-Prior")
plt.xlabel("Speed (v)")
plt.ylabel("Probability Density")
plt.title(f"Q-Prior Learned by QG with MMD ({n_qubits} qubits)")
plt.legend()
plt.savefig(f"QG_kf_{n_qubits}_mmd.png")
plt.show()

np.save(f"Q-Prior_kf_{n_qubits}_mmd.npy", global_pdf)
print(f"Global speed distribution has been saved to global_speed_pdf_kf_{n_qubits}_mmd.npy")
