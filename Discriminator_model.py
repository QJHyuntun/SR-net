import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

# 多尺度卷积判别模块
class MultiScaleConvDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleConvDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)  # 256x256 -> 128x128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)  # 128x128 -> 64x64
        self.conv3 = nn.Conv2d(128, 256, kernel_size=7, stride=2, padding=3)  # 64x64 -> 32x32

        self.residual_blocks = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )  # 保持32x32的尺度

        # 根据特征图尺寸计算全连接层的输入维度
        self.fc = nn.Linear(256 * 32 * 32, 1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  # 256x256 -> 128x128
        x2 = F.relu(self.conv2(x1))  # 128x128 -> 64x64
        x3 = F.relu(self.conv3(x2))  # 64x64 -> 32x32
        x = self.residual_blocks(x3)  # 32x32 -> 32x32
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)  # 全连接层
        return x


# 全局判别模块
class TransformerGlobalDiscriminator(nn.Module):
    def __init__(self):
        super(TransformerGlobalDiscriminator, self).__init__()
        self.transformer = create_model('swinv2_base_window8_256', pretrained=True)
        self.fc = nn.Linear(1000, 1)  # 根据你的错误信息，这里是 1000

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)  # 应用全连接层
        return x


# 层级特征判别模块
class HierarchicalFeatureDiscriminator(nn.Module):
    def __init__(self):
        super(HierarchicalFeatureDiscriminator, self).__init__()
        self.conv = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(256 * 256 * 256, 1)  # 确保这里的输入维度与展平后的特征图匹配

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x



# 综合判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.local_discriminator = MultiScaleConvDiscriminator()
        self.global_discriminator = TransformerGlobalDiscriminator()
        self.hierarchical_discriminator = HierarchicalFeatureDiscriminator()

    def forward(self, x):
        local_output = self.local_discriminator(x)
        global_output = self.global_discriminator(x)
        hierarchical_output = self.hierarchical_discriminator(x)
        return local_output, global_output, hierarchical_output


# 损失函数
class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, real_output, fake_output, real_features, fake_features):
        adversarial_loss = F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output)) + \
                           F.binary_cross_entropy_with_logits(real_output, torch.zeros_like(real_output))

        perceptual_loss = F.mse_loss(fake_features, real_features)

        total_loss = adversarial_loss + perceptual_loss
        return total_loss
