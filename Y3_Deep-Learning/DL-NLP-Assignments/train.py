import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True) 
    # Device configuration for Mac M1 (Metal Performance Shaders - MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # SVHN Dataset
    def get_svhn_loaders(batch_size=128):
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = torchvision.datasets.SVHN(root='./data', split='train', transform=transform, download=True)
        test_dataset = torchvision.datasets.SVHN(root='./data', split='test', transform=transform, download=True)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, test_loader

    train_loader, test_loader = get_svhn_loaders()

    # Define CNN Autoencoder
    class Autoencoder(nn.Module):
        def __init__(self, sparse=False):
            super(Autoencoder, self).__init__()
            self.sparse = sparse
            
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    # Training Function
    def train_autoencoder(model, train_loader, optimizer, criterion, epochs=100, sparse=False):
        model.to(device)
        
        for epoch in range(epochs):
            total_loss = 0
            for images, _ in train_loader:
                images = images.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, images)
                
                # L1 sparsity constraint
                if sparse:
                    l1_lambda = 1e-5  # Regularization strength
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss += l1_lambda * l1_norm
                
                loss.backward()
                optimizer.step()
                
                # Weight Clipping
                for p in model.parameters():
                    p.data.clamp_(-0.5, 0.5)
                
                total_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}")
        
        return model

    # PSNR Calculation
    def evaluate_psnr(model, test_loader):
        model.eval()
        total_psnr = 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                images = images.cpu().numpy()
                outputs = outputs.cpu().numpy()
                
                batch_psnr = np.mean([psnr(images[i], outputs[i]) for i in range(len(images))])
                total_psnr += batch_psnr
        
        return total_psnr / len(test_loader)

    # Train Autoencoders
    mse_autoencoder = Autoencoder(sparse=False).to(device)
    mse_sparse_autoencoder = Autoencoder(sparse=True).to(device)

    # Optimizers
    optimizer_mse = optim.Adam(mse_autoencoder.parameters(), lr=1e-3)
    optimizer_sparse = optim.Adam(mse_sparse_autoencoder.parameters(), lr=1e-3)

    # Loss function
    criterion = nn.MSELoss()

    print("Training Autoencoder with MSE Loss")
    mse_autoencoder = train_autoencoder(mse_autoencoder, train_loader, optimizer_mse, criterion, epochs=100, sparse=False)
    print("Training Autoencoder with MSE + L1 Regularization")
    mse_sparse_autoencoder = train_autoencoder(mse_sparse_autoencoder, train_loader, optimizer_sparse, criterion, epochs=100, sparse=True)

    # Evaluate Models
    psnr_mse = evaluate_psnr(mse_autoencoder, test_loader)
    psnr_sparse = evaluate_psnr(mse_sparse_autoencoder, test_loader)

    print(f"PSNR (MSE Autoencoder): {psnr_mse:.2f}")
    print(f"PSNR (MSE + L1 Sparse Autoencoder): {psnr_sparse:.2f}")

    # Display some reconstructions
    def show_images(model, test_loader):
        model.eval()
        with torch.no_grad():
            images, _ = next(iter(test_loader))
            images = images[:8].to(device)
            outputs = model(images).cpu()
            images = images.cpu()
            
            fig, axes = plt.subplots(2, 8, figsize=(15, 4))
            for i in range(8):
                axes[0, i].imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
                axes[1, i].imshow(np.transpose(outputs[i].numpy(), (1, 2, 0)))
                axes[0, i].axis('off')
                axes[1, i].axis('off')
            plt.show()

    print("Reconstructed Images: MSE Autoencoder")
    show_images(mse_autoencoder, test_loader)
    print("Reconstructed Images: MSE + L1 Sparse Autoencoder")
    show_images(mse_sparse_autoencoder, test_loader)
