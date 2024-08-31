import argparse
import os
from math import log10

import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from Generator_model import Generator, GeneratorLoss
from Discriminator_model import Discriminator, DiscriminatorLoss


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练超分辨率模型')
    parser.add_argument('--crop_size', default=256, type=int, help='训练图像裁剪大小')
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8], help='超分辨率放大倍数')
    parser.add_argument('--num_epochs', default=100, type=int, help='训练轮次')
    parser.add_argument('--total_iterations', default=200000, type=int, help='总迭代次数')
    parser.add_argument('--lr_schedule_step', default=50000, type=int, help='学习率调整步数')
    parser.add_argument('--initial_lr', default=1e-4, type=float, help='初始学习率')
    parser.add_argument('--lambda_loss', default=10, type=float, help='损失函数中的lambda值')
    parser.add_argument('--eta', default=5e-3, type=float, help='eta值')
    parser.add_argument('--gamma', default=1e-6, type=float, help='gamma值')
    opt = parser.parse_args()

    # 设置参数
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    TOTAL_ITERATIONS = opt.total_iterations
    LR_SCHEDULE_STEP = opt.lr_schedule_step
    INITIAL_LR = opt.initial_lr
    LAMBDA_LOSS = opt.lambda_loss
    ETA = opt.eta
    GAMMA = opt.gamma

    # 数据集和数据加载器
    train_set = TrainDatasetFromFolder('F:\\Download\\SRdata\\train', crop_size=CROP_SIZE,
                                       upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('F:\\Download\\SRdata\\val', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=2, batch_size=1, shuffle=False)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    generator_criterion = GeneratorLoss().to(device)
    discriminator_criterion = DiscriminatorLoss().to(device)

    # 打印模型参数数量
    print('# 生成器参数数量:', sum(param.numel() for param in netG.parameters()))
    print('# 判别器参数数量:', sum(param.numel() for param in netD.parameters()))

    # 优化器
    optimizerG = optim.Adam(netG.parameters(), lr=INITIAL_LR, betas=(0.9, 0.99))
    optimizerD = optim.Adam(netD.parameters(), lr=INITIAL_LR, betas=(0.9, 0.99))

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS}')
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            real_img = target.to(device).float()
            z = data.to(device).float()

            # 判别器训练
            fake_img = netG(z).detach()
            real_out, global_out, hierarchical_out = netD(real_img)
            fake_out, global_out_fake, hierarchical_out_fake = netD(fake_img)

            # 计算判别器损失
            d_loss = discriminator_criterion(real_out, fake_out, global_out, global_out_fake)

            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            # 生成器训练
            fake_img = netG(z)
            fake_out, global_out_fake, hierarchical_out_fake = netD(fake_img)

            # 计算生成器损失
            charbonnier_loss, perceptual_loss, adversarial_loss, tv_loss = generator_criterion(fake_img, real_img,
                                                                                               fake_out)
            g_loss = charbonnier_loss + perceptual_loss + adversarial_loss + tv_loss

            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()

            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.mean().item() * batch_size
            running_results['g_score'] += fake_out.mean().item() * batch_size

            train_bar.set_description(desc='[%d/%d] 损失_D: %.4f 损失_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        # 验证阶段
        netG.eval()
        out_path = f'training_results/SRF_{UPSCALE_FACTOR}/'
        os.makedirs(out_path, exist_ok=True)

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Validation')
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            max_val_hr = 0

            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size

                val_lr = val_lr.to(device)
                val_hr = val_hr.to(device)

                sr = netG(val_lr)

                batch_mse = ((sr - val_hr) ** 2).data.mean()
                batch_ssim = pytorch_ssim.ssim(sr, val_hr).item()

                valing_results['mse'] += batch_mse * batch_size
                valing_results['ssims'] += batch_ssim * batch_size

                max_val_hr = max(max_val_hr, val_hr.max().item())

                val_images.extend([
                    display_transform()(val_hr_restore.squeeze(0)),
                    display_transform()(val_hr.squeeze(0)),
                    display_transform()(sr.squeeze(0))
                ])

                del val_lr, val_hr, sr, val_hr_restore
                torch.cuda.empty_cache()

            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='Saving Results')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + f'epoch_{epoch}_index_{index}.png', padding=5)
                index += 1

            valing_results['psnr'] = 10 * log10(
                (max_val_hr ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
            val_bar.set_description(
                desc=f'[PSNR: {valing_results["psnr"]:.4f} dB SSIM: {valing_results["ssim"]:.4f}]')

        torch.save(netG.state_dict(), f'epochs/netG_epoch_{UPSCALE_FACTOR}_{epoch}.pth')
        torch.save(netD.state_dict(), f'epochs/netD_epoch_{UPSCALE_FACTOR}_{epoch}.pth')

        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            os.makedirs(out_path, exist_ok=True)
            data_frame = pd.DataFrame(
                data={'损失_D': results['d_loss'], '损失_G': results['g_loss'], '得分_D': results['d_score'],
                      '得分_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + f'srf_{UPSCALE_FACTOR}_train_results.csv', index_label='轮次')

        # 学习率调整
        if epoch % (TOTAL_ITERATIONS // LR_SCHEDULE_STEP) == 0:
            for param_group in optimizerG.param_groups:
                param_group['lr'] *= 0.5
            for param_group in optimizerD.param_groups:
                param_group['lr'] *= 0.5

if __name__ == '__main__':
    main()
