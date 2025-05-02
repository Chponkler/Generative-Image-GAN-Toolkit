# ┌── Parameters (Параметры)
checkpoint_path = "/content/checkpoints/ckpt_epoch_50.pth"  # path to checkpoint .pth file (путь к файлу чекпоинта .pth)
num_images      = 25                                       # number of images to generate (количество генерируемых изображений)
output_dir      = "/content/generated"                     # directory to save generated images (папка для сохранения сгенерированных)
os.makedirs(output_dir, exist_ok=True)

import torch, math, torchvision
from torch import nn

# ┌── The same dynamic Generator (Тот же динамический Generator)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # compute log2 of image size (вычисляем log2 от размера изображения)
        n = int(math.log2(image_size))
        # multiplier for feature maps (коэффициент масштабирования карт признаков)
        mult = 2**(n-3)
        layers = [
            # initial block: latent vector ➔ feature maps, output 4×4 (начальный блок: латентный вектор ➔ фичи, выход 4×4)
            nn.ConvTranspose2d(nz, ngf * mult, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * mult),
            nn.ReLU(True)
        ]
        c = ngf * mult
        # upsampling blocks until reaching image_size/2 (блоки апсемплинга до image_size/2)
        for _ in range(n-3):
            layers += [
                nn.ConvTranspose2d(c, c // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(c // 2),
                nn.ReLU(True)
            ]
            c //= 2
        # final block: feature maps ➔ 3 channels (финальный блок: фичи ➔ 3 канала RGB)
        layers += [
            nn.ConvTranspose2d(c, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # scales outputs to [-1,1] (масштабирует выходы в диапазон [-1,1])
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ┌── Load checkpoint and inference (Загрузка чекпоинта и инференс)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # choose device (выбор устройства)
G = Generator().to(device)                                             # initialize generator (инициализация генератора)
ckpt = torch.load(checkpoint_path, map_location=device)                # load checkpoint (загрузка чекпоинта)
G.load_state_dict(ckpt['G'])                                           # load weights (загрузка весов)
G.eval()                                                               # set to evaluation mode (режим оценки)

# generate random noise and produce images (генерация шума и получение изображений)
noise = torch.randn(num_images, nz, 1, 1, device=device)
with torch.no_grad():                                                  # no gradient computation (без градиентов)
    fake = G(noise).cpu()

# save each generated image (сохранение каждого сгенерированного изображения)
for i, img in enumerate(fake):
    torchvision.utils.save_image(img, f"{output_dir}/img_{i:03d}.png", normalize=True)
print(f"Saved {num_images} images to `{output_dir}`")                   # final message (итоговое сообщение)
