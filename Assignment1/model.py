import torch
import torch.nn as nn
import timm


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),  # Added BatchNorm before Sigmoid
            nn.Sigmoid()
        )

    def forward(self, x):
        attn_avg = self.global_avg_pool(x)
        attn = self.fc(attn_avg)  # Combine both attentions
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return x * self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.se = SEBlock(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.se(x)
        x = self.sa(x)
        return x


class CustomHead(nn.Module):
    def __init__(self, in_features, num_classes=100):
        super(CustomHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.head(x)


class CustomSEResNet(nn.Module):
    # only for resnet50 or resnext50 backbone
    def __init__(self, base_model, num_classes=100, freeze=True):
        super(CustomSEResNet, self).__init__()

        # Remove the final fully connected layer
        num_features = base_model.fc.in_features  # Get input features of original fc layer
        base_model.fc = nn.Identity()  # Remove the original classification layer

        self.base_model = base_model
        self.head = CustomHead(num_features, num_classes)  # Custom classification head

        self.se_layer3 = SEBlock(1024)  # layer3 的輸出通道數
        self.se_layer4 = SEBlock(2048)  # layer4 的輸出通道數

        if freeze:
            self.freeze_base_model()

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.se_layer3(self.base_model.layer3(x))  # 插入 SE 注意力
        x = self.se_layer4(self.base_model.layer4(x))  # 插入 SE 注意力

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def freeze_base_model(self):
        """Freeze all layers in the base model"""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        """Freeze all layers in the base model"""
        for param in self.base_model.parameters():
            param.requires_grad = True

    def freeze_layer3(self):
        """Freeze layer3 in the base model"""
        for name, param in self.base_model.named_parameters():
            if "layer3" in name:
                param.requires_grad = False

    def unfreeze_layer3(self):
        """Unfreeze layer3 in the base model"""
        for name, param in self.base_model.named_parameters():
            if "layer3" in name:
                param.requires_grad = True

    def unfreeze_layer4(self):
        """Unfreeze layer4 in the base model"""
        for name, param in self.base_model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True


class CustomResNet(nn.Module):
    def __init__(self, base_model, num_classes=100, freeze=True):
        super(CustomResNet, self).__init__()
        self.base_model = base_model
        num_features = base_model.fc.in_features  # 取得原本 ResNet fc 層的輸入大小
        self.base_model.fc = nn.Identity()  # 移除原本的 fc 層
        self.head = CustomHead(num_features, num_classes)  # 自訂 head

        if freeze:
            self.freeze_base_model()

    def forward(self, x):
        x = self.base_model(x)  # 先通過 ResNet 主幹
        x = self.head(x)
        return x

    def unfreeze_head(self):
        """Freeze the custom head to prevent training."""
        for param in self.head.parameters():
            param.requires_grad = True

    def freeze_base_model(self):
        """ 凍結 ResNet 的所有層，讓它變成特徵提取器 """
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        """Freeze all layers in the base model"""
        for param in self.base_model.parameters():
            param.requires_grad = True

    def freeze_layer3(self):
        for name, param in self.base_model.named_parameters():
            if "layer3" in name:
                param.requires_grad = False

    def unfreeze_layer3(self):
        for name, param in self.base_model.named_parameters():
            if "layer3" in name:
                param.requires_grad = True

    def unfreeze_layer4(self):
        for name, param in self.base_model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_type, weights_pth=None):
    pretrained_model = None
    model = None  # Ensure model is always defined

    if model_type == 'resnet50':
        pretrained_model = timm.create_model('resnet50.fb_swsl_ig1b_ft_in1k', pretrained=True)
        model = CustomResNet(pretrained_model, num_classes=100, freeze=False)

    elif model_type == 'resnet101':
        pretrained_model = timm.create_model('resnet101.a1h_in1k', pretrained=True)
        model = CustomResNet(pretrained_model, num_classes=100, freeze=False)

    elif model_type == 'resnext50_32x4d':
        pretrained_model = timm.create_model('resnext50_32x4d.fb_swsl_ig1b_ft_in1k', pretrained=True)
        model = CustomResNet(pretrained_model, num_classes=100, freeze=False)

    elif model_type == 'seresnext50_32x4d':
        pretrained_model = timm.create_model('seresnext50_32x4d.racm_in1k', pretrained=True)
        model = CustomResNet(pretrained_model, num_classes=100, freeze=False)

    elif model_type == 'seresnext101_32x8d':
        pretrained_model = timm.create_model('seresnextaa101d_32x8d.sw_in12k_ft_in1k_288', pretrained=True)
        model = CustomResNet(pretrained_model, num_classes=100, freeze=False)

    elif model_type == 'resnext101_32x4d':
        pretrained_model = timm.create_model('resnext101_32x4d.fb_swsl_ig1b_ft_in1k', pretrained=True)
        model = CustomResNet(pretrained_model, num_classes=100, freeze=False)

    elif model_type == 'resnext101_32x8d':
        pretrained_model = timm.create_model('resnext101_32x8d.fb_wsl_ig1b_ft_in1k', pretrained=True)
        model = CustomResNet(pretrained_model, num_classes=100, freeze=False)

    else:
        raise ValueError(f"Invalid model type '{model_type}'. Supported models: "
                         f"resnet50, resnet101, resnext50_32x4d, regnet_y_8gf, regnet_y_16gf.")

    if weights_pth is not None:
        model.load_state_dict(torch.load(weights_pth, map_location=device))
        print(f"Loaded model weights from {weights_pth}")

    return model
