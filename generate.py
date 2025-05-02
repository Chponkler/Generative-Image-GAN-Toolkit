# 0. Install dependencies (Установка зависимостей)
!pip install torch torchvision tqdm

# 1. Parameters (Параметры)
dataset_path      = "/content/drive/MyDrive/dataset/data300"  # path to folder with images (путь к папке с изображениями)
batch_size        = 64
image_size        = 128    # must be power of 2 (должно быть степенью двойки)
nz                = 100    # length of latent vector (длина латентного вектора)
ngf               = 64     # base feature maps in G (базовые фичи в генераторе)
ndf               = 64     # base feature maps in D (базовые фичи в дискриминаторе)
num_epochs        = 50
lr                = 0.0002
beta1             = 0.5
checkpoint_dir    = "/content/checkpoints"
save_every        = 10     # save every N epochs (сохранять каждые N эпох)
resume_checkpoint = None   # path to .pth to resume (путь к .pth для возобновления)

import os, math
import torch
from torch import nn, optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from IPython.display import display

# 2. Setup device (Настройка устройства)
os.makedirs(checkpoint_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Define a dataset for a flat folder of images (Класс датасета для «плоской» папки изображений)
class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = [
            os.path.join(root, fn)
            for fn in os.listdir(root)
            if fn.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label (заглушка метки)

# 4. Data transformations with augmentations (Трансформации данных с аугментациями)
transform = transforms.Compose([
    transforms.Resize(image_size),                             # Resize to square (Изменить размер до квадрата)
    transforms.CenterCrop(image_size),                         # Center crop (Центральное кадрирование)
    transforms.RandomHorizontalFlip(p=0.5),                    # Random horizontal flip (Случайное горизонтальное отражение)
    transforms.RandomVerticalFlip(p=0.2),                      # Random vertical flip (Случайное вертикальное отражение)
    transforms.RandomRotation(15),                             # Random rotation ±15° (Случайный поворот ±15°)
    transforms.RandomAffine(
        degrees=0,                                             # no extra rotation here (доп. поворотов нет)
        translate=(0.1, 0.1),                                  # translate up to 10% (смещение до 10%)
        scale=(0.8, 1.2),                                      # scale 80–120% (масштаб 80–120%)
        shear=10                                               # shear up to 10° (сдвиг углом до 10°)
    ),
    transforms.ColorJitter(
        brightness=0.2,                                        # brightness ±20% (яркость ±20%)
        contrast=0.2,                                          # contrast ±20% (контраст ±20%)
        saturation=0.2,                                        # saturation ±20% (насыщенность ±20%)
        hue=0.1                                                # hue ±0.1 (оттенок ±0.1)
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3), # Random perspective (Случайная перспектива)
    transforms.ToTensor(),                                     # To tensor (В тензор)
    transforms.Normalize([0.5]*3, [0.5]*3),                    # Normalize to [-1,1] (Нормализация в [-1,1])
])

# 5. Dataset and DataLoader (Набор данных и загрузчик)
dataset = FlatImageDataset(dataset_path, transform=transform)
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 6. Define dynamic Generator and Discriminator (Определение динамических Generator и Discriminator)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        n = int(math.log2(image_size))
        assert 2**n == image_size and n >= 3, "image_size must be power of 2 ≥8"
        mult = 2**(n-3)
        layers = [
            nn.ConvTranspose2d(nz, ngf*mult, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*mult),
            nn.ReLU(True)
        ]
        c = ngf*mult
        for _ in range(n-3):
            layers += [
                nn.ConvTranspose2d(c, c//2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(c//2),
                nn.ReLU(True)
            ]
            c //= 2
        layers += [
            nn.ConvTranspose2d(c, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # scales outputs to [-1,1] (масштабирует выходы в [-1,1])
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        n = int(math.log2(image_size))
        layers = [
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        c = ndf
        for _ in range(n-3):
            layers += [
                nn.Conv2d(c, c*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(c*2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            c *= 2
        layers += [
            nn.Conv2d(c, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # output probability (вероятность)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)

# 7. Initialize models, loss and optimizers (Инициализация моделей, loss и оптимайзеров)
G = Generator().to(device)
D = Discriminator().to(device)
criterion  = nn.BCELoss()  # Binary Cross Entropy (Бинарная кросс-энтропия)
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

# 8. Load checkpoint if resuming (Загрузка чекпоинта при возобновлении)
start_epoch = 1
if resume_checkpoint:
    ckpt = torch.load(resume_checkpoint, map_location=device)
    G.load_state_dict(ckpt['G'])
    D.load_state_dict(ckpt['D'])
    optimizerG.load_state_dict(ckpt['optG'])
    optimizerD.load_state_dict(ckpt['optD'])
    start_epoch = ckpt['epoch'] + 1
    print(f"Resumed from epoch {ckpt['epoch']} (Возобновлено с эпохи {ckpt['epoch']})")

fixed_noise = torch.randn(64, nz, 1, 1, device=device)  # fixed noise for visualization (фиксированный шум для визуализации)

# 9. Training loop (Цикл обучения)
for epoch in range(start_epoch, num_epochs+1):
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}")
    for real, _ in pbar:
        real = real.to(device)
        b = real.size(0)
        real_lbl = torch.ones(b, device=device)   # real label = 1 (метка «реально»)
        fake_lbl = torch.zeros(b, device=device)  # fake label = 0 (метка «фейк»)

        # Train Discriminator on real (Обучение дискриминатора на реальных)
        D.zero_grad()
        out_real = D(real)
        lossD_real = criterion(out_real, real_lbl)
        lossD_real.backward()

        # Train Discriminator on fake (Обучение дискриминатора на фейковых)
        noise = torch.randn(b, nz, 1, 1, device=device)
        fake = G(noise)
        out_fake = D(fake.detach())
        lossD_fake = criterion(out_fake, fake_lbl)
        lossD_fake.backward()
        optimizerD.step()

        # Train Generator (Обучение генератора)
        G.zero_grad()
        out = D(fake)
        lossG = criterion(out, real_lbl)
        lossG.backward()
        optimizerG.step()

        pbar.set_postfix(lossD=(lossD_real+lossD_fake).item(), lossG=lossG.item())

    # Save checkpoint (Сохранение чекпоинта)
    if epoch % save_every == 0 or epoch == num_epochs:
        path = os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch, 'G': G.state_dict(), 'D': D.state_dict(),
            'optG': optimizerG.state_dict(), 'optD': optimizerD.state_dict()
        }, path)
        print(f"Saved checkpoint: {path}")

    # Visualization (Визуализация)
    with torch.no_grad():
        imgs = G(fixed_noise).cpu()
    grid = utils.make_grid(imgs, padding=2, normalize=True)
    utils.save_image(grid, f"/content/fake_{epoch}.png")
    display(grid)
