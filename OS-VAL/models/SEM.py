import torch
import torch.nn as nn
import torch.nn.functional as F

class Semantic_Enhancement_Module(nn.Module):
    def __init__(self, in_dim=512, num_class=15):
        super().__init__()
        self.in_dim = in_dim
        self.num_class = num_class

        # 标签特征处理 (简单对齐到 in_dim)
        self.label_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(num_class, in_dim, 3, padding=1),
            nn.GroupNorm(16, in_dim),
            nn.ReLU(inplace=True)
        )

        # 通道注意力 (SE-block)
        self.se_fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 4, in_dim)
        )

        # 显著图生成 (avg+max pooling → conv)
        self.saliency_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        # 空间注意力融合 (conv + sigmoid)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)

        # 可学习融合系数
        self.alpha = nn.Parameter(torch.tensor(0.6))  # 通道
        self.beta = nn.Parameter(torch.tensor(0.6))   # 空间

    def forward(self, x, label):
        """
        输入:
          x: (B,N,C)
          label: (B,num_class,H,W)
        输出:
          enhanced_feat: (B,N,C)
        """
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        device = x.device

        if label is None:
            label = torch.zeros((B, self.num_class, 224, 224), device=device)

        # reshape (B,N,C) -> (B,C,H,W)
        x_4d = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B,C,16,16)

        # 标签特征
        label_feat = F.interpolate(label, size=(16, 16), mode="bilinear", align_corners=False)
        label_feat = self.label_processor(label_feat)  # (B,C,16,16)

        # -----------------
        # 通道注意力 (SE)
        # -----------------
        se_pool = F.adaptive_avg_pool2d(x_4d, (1, 1)).view(B, -1)  # (B,C)
        se_out = self.se_fc(se_pool).view(B, C, 1, 1)
        channel_weight = torch.sigmoid(se_out)  # (B,C,1,1)
        channel_scale = 1.0 + self.alpha * channel_weight

        # -----------------
        # 显著图生成
        # -----------------
        avg_pool = torch.mean(x_4d, dim=1, keepdim=True)   # (B,1,H,W)
        max_pool, _ = torch.max(x_4d, dim=1, keepdim=True) # (B,1,H,W)
        saliency = torch.cat([avg_pool, max_pool], dim=1)  # (B,2,H,W)
        saliency_map = torch.sigmoid(self.saliency_conv(saliency))  # (B,1,H,W)

        # -----------------
        # 空间注意力 (显著图 + 标签引导)
        # -----------------
        spatial_input = torch.cat([saliency_map, torch.mean(label_feat, dim=1, keepdim=True)], dim=1)  # (B,2,H,W)
        spatial_weight = torch.sigmoid(self.spatial_conv(spatial_input))  # (B,1,H,W)
        spatial_scale = 1.0 + self.beta * spatial_weight

        # -----------------
        # 融合
        # -----------------
        enhanced_feat_4d = x_4d * channel_scale * spatial_scale
        enhanced_feat = enhanced_feat_4d.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        return enhanced_feat

# import torch.nn as nn
# import torch
# from torch.nn.parameter import Parameter
# from torch.nn import functional as F

# class Semantic_Enhancement_Module(nn.Module):
#     def __init__(self, in_channels=512, num_class=15, groups=16):  # 修改参数顺序和默认值
#         super().__init__()
#         self.groups = groups
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#         # 调整线性层维度匹配
#         self.label_w = nn.Sequential(
#             nn.Linear(num_class, in_channels),
#             nn.ReLU(),
#             nn.Linear(in_channels, in_channels)
#         )

#         # 增加通道对齐卷积
#         self.channel_align = nn.Conv2d(in_channels, in_channels, 1)

#         # 参数初始化调整
#         self.weight = Parameter(torch.zeros(1, groups, 1, 1))
#         self.bias = Parameter(torch.ones(1, groups, 1, 1))
#         self.sig = nn.Sigmoid()

#         # 初始化参数
#         nn.init.normal_(self.label_w[0].weight, std=0.02)
#         nn.init.normal_(self.label_w[2].weight, std=0.02)

#     def forward(self, x, label):
#         # 输入形状处理
#         b, c, h, w = x.size()  # 确保输入已经是4D形状

#         # 标签处理（假设label是one-hot编码）
#         label_emb = self.label_w(label)  # (b, c)
#         label_emb = label_emb.view(b, c, 1, 1).expand_as(x)  # (b, c, h, w)

#         # 通道对齐
#         x = self.channel_align(x)

#         # 特征增强
#         x_in = x + label_emb
#         xn = x * self.avg_pool(x_in)
#         xn = xn.sum(dim=1, keepdim=True)  # (b, 1, h, w)

#         # 分组标准化
#         t = xn.view(b * self.groups, -1)
#         t = t - t.mean(dim=1, keepdim=True)
#         std = t.std(dim=1, keepdim=True) + 1e-5
#         t = t / std
#         t = t.view(b, self.groups, h, w)
#         t = t * self.weight + self.bias
#         t = t.view(b * self.groups, 1, h, w)

#         # 门控融合
#         x = x * self.sig(t)
#         return x.view(b, c, h, w)