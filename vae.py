import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
data_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

latent_dim = 64

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=latent_dim):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # Since log(var) allows -inf to 0 vals and can be reversed with exp to get var
        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def encode(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
        
    def decode(self, z):
        h = self.relu(self.fc2(z))
        out = self.sigmoid(self.fc3(h))
        return out
             
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar    
        
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    kl_loss = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1)
    return recon_loss + kl_loss

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

epochs = 5
model.train()
for epoch in range(epochs):
    total_loss = 0
    for x, _ in (train_loader):
        x = x.view(-1, 784).to(device) # (batch, channel, width,, height) to (batch, c*w*h)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        loss = vae_loss(x_recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
model.eval()
with torch.no_grad():
    x, _ = next(iter(train_loader))
    x = x.view(-1, 784).to(device)
    x_recon, _, _ = model(x)
    num_images = 10
    plt.figure(figsize=(12, 3))
    for i in range(num_images):
        # Original Image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(x[i].view(28, 28).cpu(), cmap='gray')
        plt.axis('off')
        # Reconstructed Image
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(x_recon[i].view(28, 28).cpu(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
with torch.no_grad():
    z = torch.randn(16, latent_dim).to(device)
    samples = model.decode(z).cpu()
    plt.figure(figsize=(6, 6))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(samples[i].view(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()
        