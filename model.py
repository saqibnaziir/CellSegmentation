import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import segmentation_models_pytorch as smp

# --- Begin new attention and residual modules ---
class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class EnhancedAttentionGate(nn.Module):
    """Enhanced Attention Gate with CBAM"""
    def __init__(self, F_g, F_l, F_int):
        super(EnhancedAttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # Add CBAM for enhanced attention
        self.cbam = CBAM(F_int)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        # Apply CBAM before final attention computation
        psi = self.cbam(psi)
        psi = self.psi(psi)
        return x * psi

class ResidualBlock(nn.Module):
    """Residual Block for skip connections"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)
# --- End new attention and residual modules ---

class Down(nn.Module):
    """Downscaling with maxpool then residual block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then residual block with enhanced attention gate"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            self.conv = ResidualBlock(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResidualBlock(in_channels, out_channels)
        self.attention_gate = EnhancedAttentionGate(
            F_g=in_channels // 2,
            F_l=in_channels // 2,
            F_int=in_channels // 4
        )
    def forward(self, x1, x2):
        x1 = self.conv_reduce(x1)
        x1_up = self.up(x1)
        diffY = x2.size()[2] - x1_up.size()[2]
        diffX = x2.size()[3] - x1_up.size()[3]
        x1_up = F.pad(x1_up, [diffX // 2, diffX - diffX // 2,
                             diffY // 2, diffY - diffY // 2])
        x2_att = self.attention_gate(x1_up, x2)
        x = torch.cat([x2_att, x1_up], dim=1)
        return self.conv(x)

class DecoderOutputFusion(nn.Module):
    """Fuses outputs from multiple decoder stages."""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        total_in_channels = sum(in_channels_list)
        self.conv = nn.Sequential(
            ResidualBlock(total_in_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
    def forward(self, x_list):
        base_size = x_list[0].shape[2:]
        x_upsampled = [x_list[0]]
        for i in range(1, len(x_list)):
            x_upsampled.append(
                F.interpolate(x_list[i], size=base_size, mode='bilinear', align_corners=False)
            )
        x = torch.cat(x_upsampled, dim=1)
        return self.conv(x)

class AttentionUNet(nn.Module):
    """Attention U-Net model with enhanced attention and residual blocks"""
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, bilinear=True):
        super(AttentionUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        # Initial convolution (residual block)
        self.inc = ResidualBlock(in_channels, base_channels)
        # Downsampling path
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        # Upsampling path with enhanced attention gates
        self.up1 = Up(base_channels * 16, base_channels * 8, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        # Final convolution is replaced by a fusion module
        self.out_fusion = DecoderOutputFusion(
            in_channels_list=[base_channels, base_channels * 2, base_channels * 4],
            out_channels=out_channels
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x1 = self.inc(x)          # 1 -> 64
        x2 = self.down1(x1)       # 64 -> 128
        x3 = self.down2(x2)       # 128 -> 256
        x4 = self.down3(x3)       # 256 -> 512
        x5 = self.down4(x4)       # 512 -> 1024
        d4 = self.up1(x5, x4)      # 1024 -> 512
        d3 = self.up2(d4, x3)      # 512 -> 256
        d2 = self.up3(d3, x2)      # 256 -> 128
        d1 = self.up4(d2, x1)      # 128 -> 64
        logits = self.out_fusion([d1, d2, d3])
        return logits

# def get_model(in_channels=1, out_channels=1, base_channels=64):
#     """Get Attention U-Net model"""
#     model = AttentionUNet(in_channels, out_channels, base_channels)
#     return model
def get_model(in_channels=1, out_channels=1, base_channels=None):
    """
    Returns an EfficientNet-B2 U-Net with ImageNet encoder weights.
    base_channels is ignored (kept for CLI compatibility).
    """
    model = smp.Unet(
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=out_channels,
        activation=None,          # raw logits
    )
    return model

def print_model_summary(model, input_size=(1, 256, 256)):
    """Print model summary with layer-wise information"""
    print("\nModel Summary:")
    print("=" * 80)
    summary(model, input_size)
    print("\nTotal Parameters:", sum(p.numel() for p in model.parameters()))
    print("Trainable Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("=" * 80)
    print("\nLayer-wise Information:")
    print("=" * 80)
    def count_parameters(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    def get_output_size(module, input_size):
        x = torch.randn(1, *input_size)
        try:
            with torch.no_grad():
                out = module(x)
            return out.shape[1:]
        except:
            return "N/A"
    for name, module in model.named_children():
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.Upsample, ResidualBlock, Down, Up, EnhancedAttentionGate)):
            params = count_parameters(module)
            print(f"{name:20} | Parameters: {params:,} | Output Size: {get_output_size(module, input_size)}")
    print("=" * 80)

if __name__ == "__main__":
    model = get_model()
    print_model_summary(model)