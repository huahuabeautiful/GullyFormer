import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel

# 检查可变形卷积可用性 (HybridASPP 需要)
try:
    from torchvision.ops import DeformConv2d
    deform_conv_available = True
except ImportError:
    deform_conv_available = False
    import warnings
    warnings.warn("DeformConv2d not available, using standard convolution instead")

# ========================================================== #
#   第一部分：HybridASPP 模块 (直接搬移过来，保持独立性)
# ========================================================== #
class HybridASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1, deformable_groups=4):
        super(HybridASPP, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, 1, padding=6 * rate, dilation=6 * rate, groups=dim_in, bias=False),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        if deform_conv_available:
            self.offset_conv = nn.Conv2d(
                dim_in, 2 * 3 * 3 * deformable_groups, 3, 1, padding=1, bias=True
            )
            dilation_val = max(1, 18 * rate // 12)
            padding_val = dilation_val

            torch_version = torch.__version__.split('.')
            major_version = int(torch_version[0]) if torch_version[0].isdigit() else 0
            if major_version >= 2:
                self.deform_conv = DeformConv2d(
                    dim_in, dim_out, kernel_size=3, stride=1, padding=padding_val, dilation=dilation_val, bias=True
                )
            else:
                self.deform_conv = DeformConv2d(
                    dim_in, dim_out, kernel_size=3, stride=1, padding=padding_val, dilation=dilation_val, bias=True, deformable_groups=deformable_groups
                )
            self.branch4_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
            self.branch4_relu = nn.ReLU(inplace=True)
            self.use_deform_conv = True
        else:
            dilation_val = max(1, 18 * rate // 12)
            padding_val = dilation_val
            self.branch4 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=padding_val, dilation=dilation_val, bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
            )
            self.use_deform_conv = False

        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out * 2, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out * 2, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out * 2, dim_out, 3, 1, padding=1, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        if self.use_deform_conv:
            offset = self.offset_conv(x)
            branch4 = self.deform_conv(x, offset)
            branch4 = self.branch4_bn(branch4)
            branch4 = self.branch4_relu(branch4)
        else:
            branch4 = self.branch4(x)

        global_feat = torch.mean(x, [2, 3], keepdim=True)
        global_feat = self.branch5_conv(global_feat)
        global_feat = self.branch5_bn(global_feat)
        global_feat = self.branch5_relu(global_feat)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=True)

        fused = torch.cat([branch1, branch2, branch3, branch4, global_feat], dim=1)
        output = self.fusion_conv(fused)
        return output

# ========================================================== #
#   第二部分：MiT + HybridASPP 缝合模型
# ========================================================== #
class CustomSegFormer(nn.Module):
    # 🌟 核心修改點：__init__ 中增加 use_aspp=True 參數
    def __init__(self, in_channels=3, out_channels=1, model_size="b0", pretrained_path=None, use_aspp=True):
        super(CustomSegFormer, self).__init__()

        model_name = pretrained_path if pretrained_path else f"nvidia/mit-{model_size}"

        # 1. 載入 Transformer 預訓練編碼器
        self.encoder = SegformerModel.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True
        )

        # mit-b0 的特徵通道數設定
        c1_channels = 32   # 淺層特徵 (1/4 尺度)
        c4_channels = 256  # 深層特徵 (1/32 尺度)

        # 🌟 核心修改點：控制是否啟用 HybridASPP
        self.use_aspp = use_aspp
        if self.use_aspp:
            self.aspp = HybridASPP(dim_in=c4_channels, dim_out=256, rate=1)
        else:
            # 當跳過 ASPP 時，我們可以用 nn.Identity() 完全不改變特徵，直接輸出
            # 或者用一個簡單的 1x1 卷積+BN+ReLU，保證消融試驗中非線性層數的一致性（推薦 Identity，最純粹）
            self.aspp_ablation = nn.Identity()

        # 3. 淺層特徵降維與特徵融合卷積 (DeepLabV3+ 風格)
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(c1_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.cls_conv = nn.Conv2d(256, out_channels, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        # 1. 提取多尺度特徵
        outputs = self.encoder(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        c1 = hidden_states[0] # [B, 32, H/4, W/4]
        c4 = hidden_states[3] # [B, 256, H/32, W/32]

        # 🌟 核心修改點：前向傳播路由切換
        if self.use_aspp:
            aspp_out = self.aspp(c4)       # 經過 HybridASPP 增強
        else:
            aspp_out = self.aspp_ablation(c4) # 完全跳過，直接拿 c4 輸出

        # 3. 淺層特徵降維
        low_level_feat = self.shortcut_conv(c1)

        # 4. 上採樣深層特徵對齊淺層 (此處 aspp_out 實際上就是未經增強的 c4)
        aspp_out_up = F.interpolate(
            aspp_out,
            size=low_level_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # 5. 拼接融合並解碼
        fused = torch.cat([aspp_out_up, low_level_feat], dim=1)
        x_fused = self.cat_conv(fused)

        # 6. 分類並恢復原圖解析度
        logits = self.cls_conv(x_fused)
        out = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        return out

if __name__ == "__main__":
    print("正在初始化 MiT-HybridDeepLab 缝合模型...")
    x = torch.randn((2, 3, 256, 256))
    model = CustomSegFormer(out_channels=1, model_size="b0")
    preds = model(x)
    print(f"✅ 输入图像形状: {x.shape}")
    print(f"✅ 模型输出形状: {preds.shape}")