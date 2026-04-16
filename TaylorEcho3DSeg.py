import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm(norm_type, num_channels):
    if norm_type == "batchnorm":
        return nn.BatchNorm3d(num_channels)
    elif norm_type == "instancenorm":
        return nn.InstanceNorm3d(num_channels, affine=True)
    else:
        raise ValueError(f"Unsupported normalization: {norm_type}")

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batchnorm', has_dropout=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = get_norm(norm, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = get_norm(norm, out_ch)
        self.dropout = nn.Dropout3d(p=0.2) if has_dropout else nn.Identity()
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act(self.norm2(self.conv2(x)))
        return x


class Encoder3D(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, norm='batchnorm', has_dropout=False):
        super().__init__()
        self.enc1 = ConvBlock3D(in_ch, base_ch, norm, has_dropout)
        self.down1 = nn.Conv3d(base_ch, base_ch * 2, kernel_size=(1,2,2), stride=(1,2,2))

        self.enc2 = ConvBlock3D(base_ch * 2, base_ch * 2, norm, has_dropout)
        self.down2 = nn.Conv3d(base_ch * 2, base_ch * 4, kernel_size=(1,2,2), stride=(1,2,2))

        self.enc3 = ConvBlock3D(base_ch * 4, base_ch * 4, norm, has_dropout)

    def forward(self, x):
        x1 = self.enc1(x)              # (B, C, D, H, W)
        x2 = self.enc2(self.down1(x1)) # (B, 2C, D, H/2, W/2)
        x3 = self.enc3(self.down2(x2)) # (B, 4C, D, H/4, W/4)
        return x1, x2, x3

class TaylorMotion3D(nn.Module):
    def __init__(self, in_ch, norm='batchnorm', use_second_order=True):
        super().__init__()
        self.use_second_order = use_second_order

        taylor_in_ch = in_ch * (3 if use_second_order else 2)
        self.fuse = nn.Sequential(
            nn.Conv3d(taylor_in_ch, in_ch, kernel_size=3, padding=1),
            get_norm(norm, in_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, feat):
        B, C, D, H, W = feat.shape
        f0 = feat

        v = torch.zeros_like(feat)
        if D > 2:
            v[:, :, 1:-1] = (feat[:, :, 2:] - feat[:, :, :-2]) * 0.5

        if self.use_second_order:
            a = torch.zeros_like(feat)
            if D > 2:
                a[:, :, 1:-1] = feat[:, :, 2:] - 2*feat[:, :, 1:-1] + feat[:, :, :-2]
            fuse_in = torch.cat([f0, v, a], dim=1)
        else:
            a = None
            fuse_in = torch.cat([f0, v], dim=1)

        motion_feat = self.fuse(fuse_in) + f0
        return motion_feat, {"f0": f0, "v": v, "a": a}

class FlowHead3D(nn.Module):
    def __init__(self, in_ch, norm='batchnorm'):
        super().__init__()
        self.flow = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1),
            get_norm(norm, in_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch, 2, kernel_size=3, padding=1),
        )

    def forward(self, motion_feat):
        return self.flow(motion_feat)  # (B, 2, D, h, w)

class Decoder3D(nn.Module):
    def __init__(self, base_ch=32, n_classes=4, norm='batchnorm', has_dropout=False):
        super().__init__()
        C1 = base_ch
        C2 = base_ch * 2
        C3 = base_ch * 4

        self.up2 = nn.ConvTranspose3d(C3, C2, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec2 = ConvBlock3D(C2 + C2, C2, norm, has_dropout)

        self.up1 = nn.ConvTranspose3d(C2, C1, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec1 = ConvBlock3D(C1 + C1, C1, norm, has_dropout)

        self.out_conv = nn.Conv3d(C1, n_classes, kernel_size=1)

    def forward(self, x1, x2, motion_feat):
        y = self.dec2(torch.cat([self.up2(motion_feat), x2], dim=1))
        y = self.dec1(torch.cat([self.up1(y), x1], dim=1))
        return self.out_conv(y)

class TaylorEcho3DSeg(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_classes=4,
        base_ch=32,
        normalization='batchnorm',
        has_dropout=True,
        use_second_order=True
    ):
        super().__init__()

        self.encoder = Encoder3D(
            in_ch=n_channels,
            base_ch=base_ch,
            norm=normalization,
            has_dropout=has_dropout
        )

        self.taylor = TaylorMotion3D(
            in_ch=base_ch * 4,
            norm=normalization,
            use_second_order=use_second_order
        )

        self.flow = FlowHead3D(
            in_ch=base_ch * 4,
            norm=normalization
        )

        self.decoder = Decoder3D(
            base_ch=base_ch,
            n_classes=n_classes,
            norm=normalization,
            has_dropout=has_dropout
        )

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        x1, x2, x3 = self.encoder(x)
        motion_feat, aux = self.taylor(x3)
        flow = self.flow(motion_feat)
        seg_logit = self.decoder(x1, x2, motion_feat)

        aux["motion_feat"] = motion_feat
        # return seg_logit, flow, aux
        return seg_logit
