import torch
import math
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("This model requires `timm` (pip install timm>=0.9.0)")

class ConvBlock(nn.Module):
    """
    Conv-BN-LeakyReLU with optional stride.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None):  # ✅ 修好名稱
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """
    Depth-wise 3×3 (+padding=1)  ➜  point-wise 1×1
    Allows channel expansion/reduction (in_ch → out_ch).
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.op(x)


class CoordinateAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        self.mid_channels = max(8, in_channels // reduction)
        self.in_channels = in_channels

        # 用於通道降維後再擴展的 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        self.relu = nn.ReLU()

        # 各軸注意力權重計算（兩條分支）
        self.conv_h = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        # X 軸方向 GAP（每一列做平均）：[B, C, H, 1]
        x_h = F.adaptive_avg_pool2d(x, (h, 1))

        # Y 軸方向 GAP（每一欄做平均）：[B, C, 1, W] -> [B, C, W, 1]
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)
        
        # 串接兩方向：得到 [B, C, H+W, 1]
        y = torch.cat([x_h, x_w], dim=2)
        
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        
        # 拆分 X、Y 支路
        x_h_attn, x_w_attn = torch.split(y, [h, w], dim=2)
        x_w_attn = x_w_attn.permute(0, 1, 3, 2)  # [B, C, 1, W]

        # 各自生成注意力圖（經 sigmoid）
        a_h = self.sigmoid(self.conv_h(x_h_attn))
        a_w = self.sigmoid(self.conv_w(x_w_attn))

        # 將兩方向注意力加權到輸入特徵圖上
        out = identity * a_h * a_w
        return out




class MixedAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 heads: int = 8,
                 reduction: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.d_k = in_channels // heads
        
        # ---- Q / K / V 投影 (1×1) -----------------
        self.q_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.k_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.v_proj = nn.Conv2d(in_channels, in_channels, 1)
        
        # ---- Ks：3×3 Conv 取得 span-aware key -----
        self.ks_proj = nn.Conv2d(in_channels, in_channels, 3, padding=1)

        # ---- SDConv 分支的兩個必要 Linear 層 -----
        self.sd_linear1 = nn.Conv2d(in_channels, in_channels, 1)
        self.sd_linear2 = nn.Conv2d(in_channels, in_channels, 1)

        # ---- Depthwise Separable Conv ----
        self.sd_conv = DepthwiseSeparableConv(in_channels, in_channels)
        
    def _self_attention(self, q, k, v):
        """
        q, k, v: [B, C, H, W]  (其中 C = heads * d_k)
        回傳: [B, C, H, W]
        """
        B, C, H, W = q.shape
        # 先展平成序列
        q = q.reshape(B, self.heads, self.d_k, H * W)
        k = k.reshape(B, self.heads, self.d_k, H * W)
        v = v.reshape(B, self.heads, self.d_k, H * W)

        attn = torch.einsum('bhdk,bhdm->bhkm', q, k)  # Q·K^T
        attn = attn / math.sqrt(self.d_k)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bhkm,bhdm->bhdk', attn, v)  # 乘回 V
        out = out.reshape(B, C, H, W)
        return out
    
    def _sd_conv(self, q, ks, v):
        """
        根據 (Q ⊙ Ks) 產生位置相關 gating，再做 depthwise separable conv
        q, ks, v: [B, C, H, W]
        """
        # 生成動態 gating (softmax over spatial)
        gate = torch.sigmoid(self.sd_linear1(q * ks))  # [B, C, H, W]
        v_scaled = v * gate
        out = self.sd_conv(v_scaled)
        out = self.sd_linear2(out)
        return out

    def forward(self, x):
        """
        x: [B, C, H, W]
        回傳: [B, C, H, W]  (先 concat，再壓回原通道數)
        """
        # Q / K / V / Ks
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Ks = self.ks_proj(x)

        # ----- Self-Attention 分支 ------------------
        sa_out = self._self_attention(Q, K, V)

        # ----- SDConv 分支 --------------------------
        sd_out = self._sd_conv(Q, Ks, V)

        # ----- Concatenate & Fuse -------------------
        out = torch.cat([sa_out, sd_out], dim=1)  # 輸出是 2C 通道
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        return self.relu(out + x)


class InceptionResNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # branch 1：1x1 Conv
        self.branch1 = ConvBlock(in_ch, out_ch, kernel_size=1)
        self.in_channels = in_ch

        # branch 2：1x1 -> 3x3
        self.branch2 = nn.Sequential(
            ConvBlock(in_ch, out_ch, kernel_size=1),
            ConvBlock(out_ch, out_ch, kernel_size=3, padding=1)
        )
        
        # 分支 3：1x1 -> 3x3 -> 3x3
        self.branch3 = nn.Sequential(
            ConvBlock(in_ch, out_ch, kernel_size=1),
            ConvBlock(out_ch, out_ch, kernel_size=3, padding=1),
            ConvBlock(out_ch, out_ch, kernel_size=3, padding=1)
        )

        # 綜合後的 1x1 Conv
        self.conv_after_concat = ConvBlock(out_ch * 3, in_ch, kernel_size=1)

        # 激活
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        identity = x

        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        
        # concat -> 1x1 conv
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.conv_after_concat(out)
        
        # residual
        out += identity
        return self.activation(out)


class RIPEAModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # ─── 主幹路徑 ────────────────────────────────
        self.conv1 = ConvBlock(channels, channels)
        self.incept = InceptionResNetBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
        
        # ─── 支路：MixedAttention → 1×1 壓縮 → ResidualBlock ───
        self.mixed_attn = MixedAttention(channels)   # → 2 × C
        self.residual = ResidualBlock(2 * channels)
        
        # ─── Concat 後的 Coordinate Attention ───────
        self.coord_attn = CoordinateAttention(in_channels=3 * channels)
        self.squeeze= nn.Sequential(
            nn.Conv2d(3 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        # 主幹
        main = self.conv1(x)
        main = self.incept(main)
        main = self.conv2(main)               # shape: [B, C, H, W]

        # 支路
        side = self.mixed_attn(x)             # shape: [B, 2C, H, W]
        side = self.residual(side)            # → [B, 2C, H, W]

        # Concat (主幹 + 支路)  →  [B, 3C, H, W]
        fused = torch.cat([main, side], dim=1)

        # Coordinate Attention
        fused = self.coord_attn(fused)          # 預設維持 2C 通道
        return self.squeeze(fused)             # (B,C,H,W)


class DownStage(nn.Module):
    """
    One encoder stage:
        RIPEA  → ConvBlock      (keep res)
                 ├─ DepthSep → AvgPool (↓2)
                 └─ Skip-Conv (↓2)
        concat → Conv 3×3  (reduce channels) → output
    If input channel = C      ➜ output channel = 2 × C
    If input size  = H × W    ➜ output size  = H/2 × W/2
    """
    def __init__(self, in_ch):
        super().__init__()
        mid_ch   = in_ch                 # after RIPEA / ConvBlock
        up_ch    = in_ch * 2             # after depth-sep & skip
        concat_ch= up_ch * 2             # two branches concatenated
        out_ch   = in_ch * 2             # feed to next RIPEA

        self.ripea      = RIPEAModule(in_ch)
        self.conv_block = ConvBlock(in_ch, mid_ch)

        # main branch
        self.depth_sep  = DepthwiseSeparableConv(mid_ch, up_ch)
        self.pool       = nn.AvgPool2d(2)

        # skip branch
        self.skip_conv  = nn.Conv2d(mid_ch, up_ch, 3, stride=2, padding=1, bias=False)

        # reduce after concat
        self.reduce_conv= ConvBlock(concat_ch, out_ch, kernel_size=3)

    def forward(self, x):                # (B,in_ch,H,W)
        x = self.ripea(x)                # (B,C,H,W)
        x = self.conv_block(x)           # (B,C,H,W)

        main = self.pool(self.depth_sep(x))   # (B,2C,H/2,W/2)
        skip = self.skip_conv(x)              # (B,2C,H/2,W/2)

        fused = torch.cat([main, skip], dim=1)# (B,4C,H/2,W/2)
        return self.reduce_conv(fused)        # (B,2C,H/2,W/2)


class RIPEANet(nn.Module):
    """
    Re-implementation of Fig-4 (input: 512×512×3).
    """
    def __init__(self, num_classes=5, return_feats=False):
        super().__init__()
        # ── stem ──────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),  # 512 → 256
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        # ConvBlock with stride-2 to reach 128×128×64
        self.stem_down = ConvBlock(32, 64, kernel_size=3, stride=2)          # 256 → 128

        self.return_feats = return_feats

        # ── three encoder stages ──────────────────────────────
        self.stage1 = DownStage( 64)   # 128→64,  64 →128 ch
        self.stage2 = DownStage(128)   #  64→32, 128 →256 ch
        self.stage3 = DownStage(256)   #  32→16, 256 →512 ch

        # ── classifier head ───────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # (B,512,1,1)
        self.fc          = nn.Linear(512, num_classes)

    # ---------------------------------------------------------
    def _forward_features(self, x):          # <- new helper
        x = self.stem(x)        # (B,32,256,256)
        x = self.stem_down(x)   # (B,64,128,128)
        x = self.stage1(x)      # (B,128, 64, 64)
        x = self.stage2(x)      # (B,256, 32, 32)
        x = self.stage3(x)      # (B,512, 16, 16)
        return x                # <- feature map

    def forward(self, x):
        feats = self._forward_features(x)
        if self.return_feats:
            return feats                        # (B,512,16,16)

        x = self.global_pool(feats)             # (B,512,1,1)
        x = torch.flatten(x, 1)
        return self.fc(x)                       # logits


class HybridCassavaNet(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained_eff: bool = True):
        super().__init__()
        self.ripea = RIPEANet()

        # ---- Track 1  (our custom CNN) ----------------------------
        self.ripea = RIPEANet(return_feats=True)   # ONLY features

        # ---- Track 2  (timm EfficientNet-B4) ----------------------
        #     `features_only=True` returns a *list* of feature maps.
        #     The last item is (B, 1792, H/32, W/32) → 16×16 for 512².
        self.eff = timm.create_model(
            "tf_efficientnetv2_l",
            pretrained=pretrained_eff,
            features_only=True,
            out_indices=[-1],           # just the final block
        )
        eff_ch = self.eff.feature_info.channels()[-1]   # 1792
        
        # ---- Classifier head --------------------------------------
        self.fused_ch = 512 + eff_ch
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),               # → (B, fused_ch)
            nn.Linear(self.fused_ch, num_classes)
        )

    def forward(self, x):
        # Track-1
        f1 = self.ripea(x)              # (B, 512, 16, 16)

        # Track-2   (timm returns a one-item list)
        f2 = self.eff(x)[0]             # (B, 1792, 16, 16)

        # Concatenate along channel axis
        fused = torch.cat([f1, f2], dim=1)   # (B, 512+1792, 16, 16)

        # GAP → logits
        return self.classifier(fused)


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        # EfficientNet-B4，僅取最後一層 feature map（stride=32）
        self.eff = timm.create_model(
            # "efficientnet_b4",
            "tf_efficientnetv2_xl",
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1],
        )
        eff_ch = self.eff.feature_info.channels()[-1]  # 1792

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(eff_ch, num_classes)
        )

    def forward(self, x):
        feats = self.eff(x)[0]  # (B, 1792, H/32, W/32)
        return self.classifier(feats)


# -------------------------------------------------------
# Optional test
# -------------------------------------------------------
if __name__ == "__main__":
    model = HybridCassavaNet(num_classes=5, pretrained_eff=False)
    dummy = torch.randn(4, 3, 384, 384)
    out = model(dummy)
    print(out.shape) 
