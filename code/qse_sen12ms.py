import os
import glob
import torch
import torch.nn as nn
import math
import pennylane as qml
import numpy as np
import rasterio
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# Dataset for SEN12MS-CR
# =====================================================
class SEN12MSDataset(Dataset):
    def __init__(self, cloudy_dir, clean_dir, bands="rgb", transform=None):
        """
        Args:
            cloudy_dir (str): Path to cloudy Sentinel-2 .tif
            clean_dir (str): Path to clean Sentinel-2 .tif
            bands: "all" (13 bands), "rgb" (3 bands), or list of indices [0,1,2...]
            transform: optional transforms (e.g. resize, normalization)
        """
        self.cloudy_paths = sorted(glob.glob(os.path.join(cloudy_dir, "*.tif")))
        self.clean_paths  = sorted(glob.glob(os.path.join(clean_dir, "*.tif")))
        assert len(self.cloudy_paths) > 0, f"No cloudy images found in {cloudy_dir}"
        assert len(self.cloudy_paths) == len(self.clean_paths), "Mismatch in sample count"

        self.bands = bands
        self.transform = transform

    def __len__(self):
        return len(self.cloudy_paths)

    def _load_tif(self, path):
        with rasterio.open(path) as src:
            arr = src.read()  # (bands, H, W)
            arr = torch.from_numpy(arr).float()
        return arr

    def _select_bands(self, tensor):
        if self.bands == "all":
            return tensor  # all 13
        elif self.bands == "rgb":
            return tensor[[2, 1, 0], :, :]  # BGR order → RGB
        elif isinstance(self.bands, list):
            return tensor[self.bands, :, :]
        else:
            raise ValueError("Invalid band selection")

    def __getitem__(self, idx):
        cloudy = self._load_tif(self.cloudy_paths[idx])
        clean  = self._load_tif(self.clean_paths[idx])

        cloudy = self._select_bands(cloudy)
        clean  = self._select_bands(clean)

        if self.transform:
            cloudy = self.transform(cloudy)
            clean  = self.transform(clean)

        return cloudy, clean


# =====================================================
# Positional Embeddings + Quantum SEBlock
# =====================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb_scale = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class QSEBlock(nn.Module):
    def __init__(self, channels, n_layers=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.n_qubits = int(math.ceil(math.log2(max(1, channels))))
        self.q_input_dim = 2 ** self.n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def qnode(x_vec, weights):
            qml.AmplitudeEmbedding(x_vec, wires=range(self.n_qubits), normalize=True, pad_with=0.0)
            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.RY(weights[l, q], wires=q)
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        self.qnode = qnode
        self.q_params = nn.Parameter(torch.randn(self.n_layers, self.n_qubits) * 0.01)
        self.fc_out = nn.Linear(self.n_qubits, channels)
        self.act = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        if c < self.q_input_dim:
            pad = self.q_input_dim - c
            y_pad = torch.cat([y, torch.zeros(b, pad, device=y.device, dtype=y.dtype)], dim=1)
        else:
            y_pad = y
        q_out = []
        for i in range(b):
            vec_cpu = y_pad[i].detach().to("cpu")
            vec_cpu = vec_cpu.requires_grad_(True)
            z = self.qnode(vec_cpu, self.q_params)
            q_out.append(torch.stack(z).to(y.device))
        q_out = torch.stack(q_out, dim=0).to(x.dtype)
        y_excite = self.act(self.fc_out(q_out))
        y_excite = y_excite.view(b, c, 1, 1)
        return x * y_excite


# =====================================================
# U-Net with QSE
# =====================================================
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.qse = QSEBlock(out_ch, n_layers=2)
        self.act2 = nn.SiLU()

    def forward(self, x, t):
        x = self.conv1(x)
        x = self.bn1(x)
        t_emb = self.time_mlp(t)
        t_emb = t_emb[:, :, None, None]
        x = x + t_emb
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.qse(x)
        x = self.act2(x)
        return x


class CloudDenoiser(nn.Module):
    def __init__(self, in_channels=3, base_dim=64):
        super().__init__()
        self.time_mlp = SinusoidalPositionEmbeddings(base_dim)

        self.down1 = UNetBlock(in_channels, base_dim, base_dim)
        self.down2 = UNetBlock(base_dim, base_dim * 2, base_dim)
        self.down3 = UNetBlock(base_dim * 2, base_dim * 4, base_dim)

        self.bottleneck = UNetBlock(base_dim * 4, base_dim * 8, base_dim)

        self.up1 = UNetBlock(base_dim * 8 + base_dim * 4, base_dim * 4, base_dim)
        self.up2 = UNetBlock(base_dim * 4 + base_dim * 2, base_dim * 2, base_dim)
        self.up3 = UNetBlock(base_dim * 2 + base_dim, base_dim, base_dim)

        self.final_conv = nn.Conv2d(base_dim, in_channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x1 = self.down1(x, t)
        x2 = self.down2(nn.MaxPool2d(2)(x1), t)
        x3 = self.down3(nn.MaxPool2d(2)(x2), t)
        x4 = self.bottleneck(nn.MaxPool2d(2)(x3), t)

        x = nn.Upsample(scale_factor=2, mode='nearest')(x4)
        x = self.up1(torch.cat([x, x3], dim=1), t)
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = self.up2(torch.cat([x, x2], dim=1), t)
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = self.up3(torch.cat([x, x1], dim=1), t)

        return self.final_conv(x)


# =====================================================
# Training + Metrics
# =====================================================
class DiffusionProcess:
    def __init__(self, T=1000):
        self.T = T


def evaluate_metrics(pred, target):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    return {
        'PSNR': psnr(pred, target, data_range=1.0).item(),
        'SSIM': ssim(pred, target, data_range=1.0).item()
    }


def visualize_outputs(noisy, pred, target, epoch, save_dir='visuals_qseb'):
    os.makedirs(save_dir, exist_ok=True)
    idx = 0
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axs, [noisy, pred, target], ['Cloudy', 'Denoised', 'Ground Truth']):
        img = img[idx].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/epoch_{epoch}.png')
    plt.close()


def train(model, diffuser, train_loader, val_loader, num_epochs=50, save_path='qse_best_model.pth'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_psnr = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for noisy_images, gt_images in train_loader:
            noisy_images = noisy_images.to(device)
            gt_images = gt_images.to(device)
            t = torch.randint(0, diffuser.T, (noisy_images.size(0),), device=device)

            target_noise = noisy_images - gt_images
            pred_noise = model(noisy_images, t)
            loss = nn.MSELoss()(pred_noise, target_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] Training Loss: {avg_loss:.4f}")

        model.eval()
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for noisy_images, gt_images in val_loader:
                noisy_images = noisy_images.to(device)
                gt_images = gt_images.to(device)
                t = torch.randint(0, diffuser.T, (noisy_images.size(0),), device=device)
                pred_noise = model(noisy_images, t)
                denoised = (noisy_images - pred_noise).clamp(0, 1)
                metrics = evaluate_metrics(denoised, gt_images)
                psnr_list.append(metrics['PSNR'])
                ssim_list.append(metrics['SSIM'])
            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_ssim = sum(ssim_list) / len(ssim_list)
            print(f"[Epoch {epoch}] PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model (PSNR: {avg_psnr:.2f})")

            visualize_outputs(noisy_images, denoised, gt_images, epoch)


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    print("Using device:", device)

    # Example paths (replace with your dataset paths!)
    train_cloud_dir = "../data/SEN12MS/spring/S2cloudy"
    train_label_dir = "../data/SEN12MS/spring/S2clear"
    val_cloud_dir   = "../data/SEN12MS/summer/S2cloudy"
    val_label_dir   = "../data/SEN12MS/summer/S2clear"

    # Dataset & Loader
    train_dataset = SEN12MSDataset(train_cloud_dir, train_label_dir, bands="rgb")
    val_dataset   = SEN12MSDataset(val_cloud_dir, val_label_dir, bands="rgb")

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Model
    model = CloudDenoiser(in_channels=3).to(device)
    diffuser = DiffusionProcess(T=1000)

    # Test run
    model.eval()
    with torch.no_grad():
        sample_input = torch.randn(1, 3, 128, 128).to(device)
        sample_t = torch.randint(0, 1000, (1,), device=device)
        output = model(sample_input, sample_t)
        print(f"Input shape: {sample_input.shape}, Output shape: {output.shape}")

    # Training
    # train(model, diffuser, train_loader, val_loader, num_epochs=50)
