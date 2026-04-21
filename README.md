import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --------
# 1. Hyperparameters
# -------------------------------
T = 1000  # total timesteps
beta_start = 1e-4
beta_end = 0.02
batch_size = 8

# -------------------------------
# 2. Noise Schedule
# -------------------------------
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1. - betas
alpha_hat = torch.cumprod(alphas, dim=0)

# -------------------------------
# 3. Dataset (Fashion-MNIST)
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
# 4. Forward Diffusion Function
# -------------------------------
def add_noise(x, t):
    """
    x: [B, 1, 28, 28]
    t: [B] timestep
    """
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]

    noise = torch.randn_like(x)
    x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
    return x_t, noise

# -------------------------------
# 5. Get Sample Images
# -------------------------------
images, labels = next(iter(loader))

# Random timesteps for each image
t = torch.randint(0, T, (images.shape[0],))

# Add noise
noisy_images, noise = add_noise(images, t)

# -------------------------------
# 6. Visualization
# -------------------------------
fig, axs = plt.subplots(2, batch_size, figsize=(15, 4))

for i in range(batch_size):
    # Original
    axs[0, i].imshow(images[i].squeeze(), cmap='gray')
    axs[0, i].set_title(f"Original")
    axs[0, i].axis('off')

    # Noisy
    axs[1, i].imshow(noisy_images[i].detach().squeeze(), cmap='gray')
    axs[1, i].set_title(f"t={t[i].item()}")
    axs[1, i].axis('off')

plt.tight_layout()
plt.show()
