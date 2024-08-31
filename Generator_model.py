import torch
import torch.nn as nn
from torchvision.models import vgg19


# 多尺度卷积特征提取模块
class MultiScaleConvFeatureExtraction(nn.Module):
    def __init__(self):
        super(MultiScaleConvFeatureExtraction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding=1)
        self.flex_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.concat_conv = nn.Conv2d(in_channels=64 + 128 + 256 + 64, out_channels=64, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        flex_conv = self.flex_conv(x)
        concat_features = torch.cat([conv1, conv2, conv3, flex_conv], dim=1)
        multi_scale_features = self.concat_conv(concat_features)
        return multi_scale_features


# 动态特征融合模块
class DynamicFeatureFusion(nn.Module):
    def __init__(self):
        super(DynamicFeatureFusion, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

    def forward(self, x):
        N, C, H, W = x.size()
        x_flat = x.reshape(N, C, H * W).permute(2, 0, 1)  # (H*W, N, C)
        attn_output, _ = self.self_attention(x_flat, x_flat, x_flat)
        attn_output = attn_output.permute(1, 0, 2).reshape(N, C, H, W)
        fused_features = self.conv(x + attn_output)
        return fused_features


# 自定义 Transformer 模块
class CustomTransformer(nn.Module):
    def __init__(self, input_dim=64, d_model=64, nhead=8, max_seq_length=4096):
        super(CustomTransformer, self).__init__()
        self.linear = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        self.layer_norm = nn.LayerNorm(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, batch_first=True)

    def forward(self, src, tgt):
        src = self.linear(src) + self.positional_encoding[:, :src.size(1)]
        tgt = self.linear(tgt) + self.positional_encoding[:, :tgt.size(1)]
        src = self.layer_norm(src)
        tgt = self.layer_norm(tgt)
        return self.transformer(src, tgt)


class MultiStageHybridTransformer(nn.Module):
    def __init__(self, input_dim=64, d_model=64):
        super(MultiStageHybridTransformer, self).__init__()
        # 创建三个阶段的 Transformer，每个阶段的 d_model 需要与输入特征通道数一致
        self.stage1_transformer = CustomTransformer(input_dim=input_dim, d_model=d_model)
        self.stage2_transformer = CustomTransformer(input_dim=d_model, d_model=d_model)
        self.stage3_transformer = CustomTransformer(input_dim=d_model, d_model=d_model)

        # 反卷积层用于将图像放大 2 倍
        self.upsample1 = nn.ConvTranspose2d(in_channels=d_model,
                                            out_channels=d_model,
                                            kernel_size=4,
                                            stride=2,
                                            padding=1,
                                            output_padding=0)
        self.upsample2 = nn.ConvTranspose2d(in_channels=d_model,
                                            out_channels=d_model,
                                            kernel_size=4,
                                            stride=2,
                                            padding=1,
                                            output_padding=0)

    def forward(self, x):
        N, C, H, W = x.shape
        assert C == self.stage1_transformer.transformer.d_model, "Feature dimension must match d_model"

        # Stage 1: 处理低分辨率特征
        x_flat = x.reshape(N, -1, C)  # [N, H*W, C]
        x_low_res = self.stage1_transformer(x_flat, x_flat)

        # 将特征从 [N, H*W, C] 转换回 [N, C, H, W]
        x_low_res = x_low_res.reshape(N, C, H, W)

        # Stage 2: 放大 2 倍
        x_medium_res = self.upsample1(x_low_res)  # [N, C, 2*H, 2*W]
        x_medium_res_flat = x_medium_res.reshape(N, -1, C)
        x_medium_res = self.stage2_transformer(x_medium_res_flat, x_medium_res_flat)

        # 将特征从 [N, 2*H*2*W, C] 转换回 [N, C, 2*H, 2*W]
        x_medium_res = x_medium_res.reshape(N, C, 2*H, 2*W)

        # Stage 3: 再次放大 2 倍
        x_high_res = self.upsample2(x_medium_res)  # [N, C, 4*H, 4*W]
        x_high_res_flat = x_high_res.reshape(N, -1, C)
        x_high_res = self.stage3_transformer(x_high_res_flat, x_high_res_flat)

        # 将特征从 [N, 4*H*4*W, C] 转换回 [N, C, 4*H, 4*W]
        x_high_res = x_high_res.reshape(N, C, 4*H, 4*W)
        print('x_high_res', x_high_res.shape)

        return x_high_res



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.feature_extraction = MultiScaleConvFeatureExtraction()
        self.feature_fusion = DynamicFeatureFusion()
        self.hybrid_transformer = MultiStageHybridTransformer()

        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=3)  # 输出RGB图像
        )

    def forward(self, x):
        features = self.feature_extraction(x)
        fused_features = self.feature_fusion(features)
        print('fused_features', fused_features.shape)
        refined_features = self.hybrid_transformer(fused_features)

        output_image = self.upsample(refined_features.reshape(refined_features.size(0), 64, 256, 256))
        return output_image


# 生成器损失函数
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.charbonnier_loss = nn.SmoothL1Loss()
        self.vgg19 = vgg19(pretrained=True).features.eval()
        for param in self.vgg19.parameters():
            param.requires_grad = False
        self.perceptual_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCELoss()
        self.total_variation_loss = self.total_variation

    def forward(self, sr_image, hr_image, real_or_fake):
        charbonnier_loss = self.charbonnier_loss(sr_image, hr_image)
        sr_features = self.vgg19(sr_image)
        hr_features = self.vgg19(hr_image)
        perceptual_loss = self.perceptual_loss(sr_features, hr_features)

        # Ensure shapes match
        sr_image_flat = sr_image.view(sr_image.size(0), -1)
        real_or_fake_flat = real_or_fake.view(real_or_fake.size(0), -1)
        adversarial_loss = self.adversarial_loss(sr_image_flat, real_or_fake_flat)

        tv_loss = self.total_variation_loss(sr_image)
        return charbonnier_loss, perceptual_loss, adversarial_loss, tv_loss

    def total_variation(self, x):
        loss = torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + torch.sum(
            torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        return loss


