# model.py  -- updated: ASPP integrated into bottleneck
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import segmentation_models_pytorch as smp

# ------------------------------------------------------------------
# 1) Attention modules (CBAM etc.)
# ------------------------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg, mx], dim=1)
        return self.sigmoid(self.conv(y))

class ChannelAttention(nn.Module):
    def __init__(self, in_ch, ratio=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, max(1, in_ch//ratio), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_ch//ratio), in_ch, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.mlp(self.avg(x))
        mx  = self.mlp(self.max(x))
        return self.sigmoid(avg + mx)

class CBAM(nn.Module):
    def __init__(self, in_ch, ratio=16, ks=7):
        super().__init__()
        self.ca = ChannelAttention(in_ch, ratio)
        self.sa = SpatialAttention(ks)
    def forward(self, x):
        x = x * self.ca(x)
        return x * self.sa(x)

class EnhancedAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, dropout=0.3):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.Wx = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.cbam = CBAM(F_int)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
    def forward(self, g, x):
        psi = self.relu(self.Wg(g) + self.Wx(x))
        psi = self.cbam(psi)
        psi = self.dropout(psi)
        psi = self.psi(psi)
        return x * psi

# ------------------------------------------------------------------
# 2) Encoder(s)
# ------------------------------------------------------------------
class EfficientNetB6Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = smp.encoders.get_encoder(
            "efficientnet-b6",
            in_channels=1,
            weights="imagenet" if pretrained else None
        )
        self.out_channels = self.backbone.out_channels
        self.encoder_name = f"EfficientNet-B6 ({'Pretrained' if pretrained else 'Random Init'})"
    def forward(self, x):
        return self.backbone(x)   # list of feature maps

# ------------------------------------------------------------------
# Residual + SE blocks for Custom Encoder (unchanged structure)
# ------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1, use_se=False):
        super().__init__()
        pad = dilation if dilation > 1 else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=pad,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=pad,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_ch, max(1, out_ch // 16), kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(1, out_ch // 16), out_ch, kernel_size=1),
                nn.Sigmoid()
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se:
            se_weight = self.se(out)
            out = out * se_weight
        out += identity
        return self.relu(out)

class CustomEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.stage1 = nn.Sequential( ResidualBlock(in_channels, 32) )
        self.stage2 = nn.Sequential( nn.MaxPool2d(2), ResidualBlock(32, 64) )
        self.stage3 = nn.Sequential( nn.MaxPool2d(2), ResidualBlock(64, 128) )
        self.stage4 = nn.Sequential( nn.MaxPool2d(2), ResidualBlock(128, 256, dilation=2) )
        self.stage5 = nn.Sequential( nn.MaxPool2d(2), ResidualBlock(256, 512, dilation=4, use_se=True) )
        self.out_channels = [32, 64, 128, 256, 512]
        self.encoder_name = "Custom Residual+Dilated+SE Encoder"
    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        return [f1, f2, f3, f4, f5]

# ------------------------------------------------------------------
# 2.5) ASPP Module (compact)
# ------------------------------------------------------------------
class ASPP(nn.Module):
    """
    Compact ASPP: parallel convs with different dilation rates + image pooling.
    Projects output back to in_ch.
    """
    def __init__(self, in_ch, out_ch=None, rates=(1,6,12,18)):
        super().__init__()
        if out_ch is None:
            out_ch = max(16, in_ch // 4)
        self.branches = nn.ModuleList()
        # 1x1 conv branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ))
        # dilated branches
        for r in rates[1:]:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
        # image pooling branch
        self.branches.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ))
        # projection to original channels
        total_br = len(self.branches)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * total_br, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        res = []
        for i, b in enumerate(self.branches):
            y = b(x)
            if i == len(self.branches)-1:  # image pooling branch -> upsample
                y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
            res.append(y)
        x = torch.cat(res, dim=1)
        return self.project(x)

# ------------------------------------------------------------------
# 3) Attention Decoder - works with 5-stage encoder
# ------------------------------------------------------------------
class AttentionDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=(256,128,64,32), dropout=0.3):
        super().__init__()
        enc_ch = encoder_channels[::-1]  # deepest first
        self.up_layers = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i, out_ch in enumerate(decoder_channels):
            in_ch = enc_ch[i] if i == 0 else decoder_channels[i-1]
            skip_ch = enc_ch[i+1] if i+1 < len(enc_ch) else 0
            # upsample
            self.up_layers.append(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2))
            # attention on skip
            if skip_ch:
                self.att_gates.append(EnhancedAttentionGate(out_ch, skip_ch, max(8, out_ch//2), dropout=dropout))
            else:
                self.att_gates.append(None)
            conv_in = out_ch + skip_ch if skip_ch else out_ch
            self.convs.append(nn.Sequential(
                nn.Conv2d(conv_in, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ))
        self.final = nn.Conv2d(decoder_channels[-1], 1, 1)

    def forward(self, feats, target_size=None):
        x = feats[-1]
        for i, (up, att, conv) in enumerate(zip(self.up_layers, self.att_gates, self.convs)):
            x = up(x)
            if att is not None:
                skip = feats[-(i+2)]
                skip = att(x, skip)
                x = torch.cat([x, skip], dim=1)
            x = conv(x)
        x = self.final(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

# ------------------------------------------------------------------
# 4) Full model (integrates ASPP into bottleneck)
# ------------------------------------------------------------------
class AttentionUnet(nn.Module):
    def __init__(self, encoder_type="efficientnet", pretrained=True, use_aspp=True):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "efficientnet":
            self.encoder = EfficientNetB6Encoder(pretrained=pretrained)
        elif encoder_type == "custom":
            self.encoder = CustomEncoder(in_channels=1)
        else:
            raise ValueError("Unknown encoder_type. Choose 'efficientnet' or 'custom'.")

        # Use ASPP on the bottleneck if requested
        final_ch = self.encoder.out_channels[-1]
        self.use_aspp = use_aspp
        if self.use_aspp:
            # keep projection back to final_ch to keep decoder unchanged
            self.aspp = ASPP(in_ch=final_ch, out_ch=max(16, final_ch//4))
        else:
            self.aspp = nn.Identity()

        self.decoder = AttentionDecoder(self.encoder.out_channels)
        self.pretrained = pretrained if encoder_type == "efficientnet" else False

    def forward(self, x):
        feats = self.encoder(x)
        # Replace last feature map with ASPP-processed version
        feats = list(feats)
        feats[-1] = self.aspp(feats[-1])
        out = self.decoder(feats, target_size=x.shape[2:])
        return out


# ------------------------------------------------------------------
# 5) Summary functions
# ------------------------------------------------------------------
def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_layer_info(module, layer_name=""):
    """Extract layer information with input/output channels"""
    layer_info = []
    
    if isinstance(module, nn.Conv2d):
        layer_info.append({
            'name': layer_name,
            'type': 'Conv2d',
            'in_channels': module.in_channels,
            'out_channels': module.out_channels,
            'kernel_size': module.kernel_size,
            'stride': module.stride,
            'padding': module.padding,
            'params': sum(p.numel() for p in module.parameters())
        })
    elif isinstance(module, nn.ConvTranspose2d):
        layer_info.append({
            'name': layer_name,
            'type': 'ConvTranspose2d',
            'in_channels': module.in_channels,
            'out_channels': module.out_channels,
            'kernel_size': module.kernel_size,
            'stride': module.stride,
            'padding': module.padding,
            'params': sum(p.numel() for p in module.parameters())
        })
    elif isinstance(module, nn.BatchNorm2d):
        layer_info.append({
            'name': layer_name,
            'type': 'BatchNorm2d',
            'in_channels': module.num_features,
            'out_channels': module.num_features,
            'kernel_size': 'N/A',
            'stride': 'N/A',
            'padding': 'N/A',
            'params': sum(p.numel() for p in module.parameters())
        })
    elif isinstance(module, nn.Linear):
        layer_info.append({
            'name': layer_name,
            'type': 'Linear',
            'in_channels': module.in_features,
            'out_channels': module.out_features,
            'kernel_size': 'N/A',
            'stride': 'N/A',
            'padding': 'N/A',
            'params': sum(p.numel() for p in module.parameters())
        })
    
    return layer_info

def print_detailed_layers(model_part, part_name):
    """Print detailed layer information"""
    print(f"\n📋 {part_name.upper()} LAYER DETAILS:")
    print("-" * 80)
    print(f"{'Layer Name':<35} {'Type':<15} {'In Ch':<8} {'Out Ch':<8} {'Kernel':<10} {'Params':<12}")
    print("-" * 80)
    
    layer_count = 0
    total_params = 0
    
    for name, module in model_part.named_modules():
        if name == "":  # Skip the root module
            continue
            
        layer_info = get_layer_info(module, name)
        for info in layer_info:
            layer_count += 1
            total_params += info['params']
            
            # Truncate long names
            display_name = info['name'][:34] if len(info['name']) > 34 else info['name']
            
            print(f"{display_name:<35} {info['type']:<15} {info['in_channels']:<8} {info['out_channels']:<8} "
                  f"{str(info['kernel_size']):<10} {info['params']:<12,}")
    
    print("-" * 80)
    print(f"Total {part_name} layers: {layer_count}")
    print(f"Total {part_name} parameters: {total_params:,}")
    return layer_count, total_params

def print_model_summary(model, input_size=(1, 256, 256), device='cuda'):
    """Print detailed summary of the model including encoder and decoder"""
    print("=" * 100)
    print("COMPREHENSIVE MODEL ARCHITECTURE SUMMARY")
    print("=" * 100)
    
    # Get encoder information dynamically
    encoder_name = getattr(model.encoder, 'encoder_name', f"{model.encoder_type.capitalize()} Encoder")
    
    # Overall model summary
    print("\n🔹 FULL MODEL SUMMARY:")
    print("-" * 50)
    try:
        summary(model, input_size, device=device)
    except Exception as e:
        print(f"Could not generate torchsummary for full model: {e}")
    
    # Total parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\n📊 PARAMETER COUNT:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Encoder detailed analysis
    print(f"\n🔹 ENCODER ANALYSIS ({encoder_name}):")
    print("-" * 50)
    encoder_params, encoder_trainable = count_parameters(model.encoder)
    print(f"   Encoder parameters: {encoder_params:,}")
    print(f"   Encoder trainable: {encoder_trainable:,}")
    print(f"   Output channels per stage: {model.encoder.out_channels}")
    
    # Encoder layers detail
    enc_layer_count, enc_params = print_detailed_layers(model.encoder, "encoder")
    
    # Decoder detailed analysis  
    print(f"\n🔹 DECODER ANALYSIS (Attention Decoder):")
    print("-" * 50)
    decoder_params, decoder_trainable = count_parameters(model.decoder)
    print(f"   Decoder parameters: {decoder_params:,}")
    print(f"   Decoder trainable: {decoder_trainable:,}")
    
    # Decoder structure info
    print(f"   Upsampling stages: {len(model.decoder.up_layers)}")
    print(f"   Attention gates: {sum(1 for gate in model.decoder.att_gates if gate is not None)}")
    
    # Decoder layers detail
    dec_layer_count, dec_params = print_detailed_layers(model.decoder, "decoder")
    
    # MODIFIED: Decoder stage-wise breakdown for 5-stage encoder
    print(f"\n📋 DECODER STAGE BREAKDOWN:")
    print("-" * 60)
    print(f"{'Stage':<8} {'Upsample':<12} {'Skip Ch':<10} {'Output Ch':<12} {'Has Attention':<15}")
    print("-" * 60)
    
    enc_ch = model.encoder.out_channels[::-1]  # deepest first
    decoder_channels = [256, 128, 64, 32]  # MODIFIED: Updated for 5-stage encoder
    
    for i, out_ch in enumerate(decoder_channels):
        in_ch = enc_ch[i] if i == 0 else decoder_channels[i-1]
        skip_ch = enc_ch[i+1] if i+1 < len(enc_ch) else 0
        has_attention = "Yes" if skip_ch > 0 else "No"
        
        print(f"Stage {i+1:<3} {in_ch:<12} {skip_ch:<10} {out_ch:<12} {has_attention:<15}")
    
    print("-" * 60)
    
    # Summary statistics
    print(f"\n🔹 ARCHITECTURE SUMMARY:")
    print("-" * 50)
    print(f"   Total model layers: {enc_layer_count + dec_layer_count}")
    print(f"   Encoder layers: {enc_layer_count}")
    print(f"   Decoder layers: {dec_layer_count}")
    print(f"   Encoder/Decoder ratio: {enc_layer_count/dec_layer_count:.1f}:1")
    print(f"   Parameter distribution: Encoder {encoder_params/total_params*100:.1f}% | Decoder {decoder_params/total_params*100:.1f}%")
    print(f"   Architecture: {encoder_name} + Attention U-Net Decoder")
    print(f"   Input: {input_size[0]} channel, {input_size[1]}×{input_size[2]} images")
    print(f"   Output: 1 channel binary segmentation masks")
    print(f"   Max encoder channels: 512 (reduced from 1024)")  # NEW: Added info about reduction
    
    print("=" * 100)

# ------------------------------------------------------------------
# 6) main
# ------------------------------------------------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test both encoders
    print("\n" + "="*50)
    print("TESTING EFFICIENTNET ENCODER")
    print("="*50)
    net_eff = AttentionUnet(encoder_type="efficientnet", pretrained=True).to(device)
    print_model_summary(net_eff, input_size=(1, 256, 256), device=device)
    
    print("\n" + "="*50)
    print("TESTING CUSTOM ENCODER (MODIFIED - 5 STAGES, MAX 512 CHANNELS)")
    print("="*50)
    net_custom = AttentionUnet(encoder_type="custom", pretrained=False, use_aspp=True).to(device)
    print_model_summary(net_custom, input_size=(1, 256, 256), device=device)
    
    # Test forward pass for both
    print(f"\n🔹 TESTING FORWARD PASSES:")
    print("-" * 50)
    with torch.no_grad():
        test_input = torch.randn(2, 1, 256, 256).to(device)
        
        output_eff = net_eff(test_input)
        output_custom = net_custom(test_input)
        
        print(f"   Input shape: {test_input.shape}")
        print(f"   EfficientNet output shape: {output_eff.shape}")
        print(f"   Custom output shape: {output_custom.shape}")
        print("   ✅ Both forward passes successful!")
        
    # ADDED: Show parameter comparison
    print(f"\n🔹 PARAMETER COMPARISON:")
    print("-" * 50)
    eff_params, _ = count_parameters(net_eff)
    custom_params, _ = count_parameters(net_custom)
    print(f"   EfficientNet model: {eff_params:,} parameters")
    print(f"   Custom model (5-stage): {custom_params:,} parameters")
    print(f"   Custom model is {eff_params/custom_params:.1f}x smaller than EfficientNet")
