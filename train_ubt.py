import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from Generator_model import Generator,GeneratorLoss
from Discriminator_model import Discriminator,DiscriminatorLoss


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.image_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in '{data_path}'")
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_path, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image  # 返回图像和标签（图像自回归）


# 实例化模型
generator = Generator()
discriminator = Discriminator()


# 优化器
def get_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))


# 学习率调度器
def get_scheduler(optimizer, step_size):
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)


# 实例化模型和优化器
def initialize_models(lr):
    g = Generator()
    d = Discriminator()
    optimizer_g = get_optimizer(g, lr)
    optimizer_d = get_optimizer(d, lr)
    scheduler_g = get_scheduler(optimizer_g, step_size=50000)
    scheduler_d = get_scheduler(optimizer_d, step_size=50000)
    return g, d, optimizer_g, optimizer_d, scheduler_g, scheduler_d


# 损失函数
generator_loss_fn = GeneratorLoss()
discriminator_loss_fn = DiscriminatorLoss()


# 训练函数
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型和优化器
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.99))

    scaler_g = GradScaler()
    scaler_d = GradScaler()

    def adjust_lr(optimizer, epoch, initial_lr):
        lr = initial_lr * (0.5 ** (epoch // 50000))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    train_dataset = CustomDataset(data_path=args.train_data)
    val_dataset = CustomDataset(data_path=args.val_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    for epoch in range(args.num_epochs):
        generator.train()
        discriminator.train()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epochs}', unit='batch')

        for i, (real_images, _) in enumerate(pbar):
            real_images = real_images.to(device)

            # 训练判别器
            optimizer_d.zero_grad()
            with autocast():
                real_output = discriminator(real_images)
                fake_images = generator(real_images)
                fake_output = discriminator(fake_images.detach())

                real_features = torch.ones_like(real_output, device=device)
                fake_features = torch.zeros_like(fake_output, device=device)
                d_loss = discriminator_loss_fn(real_output, fake_output, real_features, fake_features)

            scaler_d.scale(d_loss).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            # 训练生成器
            optimizer_g.zero_grad()
            with autocast():
                fake_images = generator(real_images)
                fake_output = discriminator(fake_images)

                charbonnier_loss, perceptual_loss, adversarial_loss, tv_loss = generator_loss_fn(fake_images, real_images, fake_output)
                g_loss = args.lambda_total * charbonnier_loss + perceptual_loss + args.eta * adversarial_loss + args.gamma * tv_loss

            scaler_g.scale(g_loss).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            # 更新进度条
            pbar.set_postfix({'G Loss': g_loss.item(), 'D Loss': d_loss.item()})

        # 学习率调度
        adjust_lr(optimizer_g, epoch, args.lr)
        adjust_lr(optimizer_d, epoch, args.lr)

        # 每50000次迭代保存模型
        if (epoch + 1) % 50000 == 0:
            torch.save(generator.state_dict(), os.path.join(args.output_dir, f'generator_epoch_{epoch + 1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f'discriminator_epoch_{epoch + 1}.pth'))

    # 输出参数数量
    print(f'Generator Parameters: {sum(p.numel() for p in generator.parameters())}')
    print(f'Discriminator Parameters: {sum(p.numel() for p in discriminator.parameters())}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN with progress tracking and mixed precision')
    parser.add_argument('--train_data', type=str, default='F:\\Download\\SRdata\\train\\', help='Path to training dataset')
    parser.add_argument('--val_data', type=str, default='F:\\Download\\SRdata\\val\\', help='Path to validation dataset')
    parser.add_argument('--output_dir', type=str, default='E:\\Super-Resolution-model\\Super-Resolution', help='Directory to save model checkpoints')
    parser.add_argument('--num_epochs', type=int, default=200000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-6, help='Initial learning rate')
    parser.add_argument('--lambda_total', type=float, default=10, help='Lambda for total loss')
    parser.add_argument('--eta', type=float, default=5e-3, help='Eta for adversarial loss')
    parser.add_argument('--gamma', type=float, default=1e-6, help='Gamma for total variation loss')

    args = parser.parse_args()
    train(args)
