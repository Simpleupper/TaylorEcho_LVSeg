import torch
from torch import nn
import torch.nn.functional as F
from math import factorial


# ===========================
# 3D Version Basic Modules
# ===========================

class ResB3D(nn.Module):
    """
    3D Residual Block: Conv3d -> LeakyReLU -> Conv3d -> LeakyReLU -> Conv3d + shortcut
    """
    def __init__(self, in_c, out_c):
        super(ResB3D, self).__init__()
        self.res = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=1, stride=1, padding=0),
        )
        self.conv_short = nn.Conv3d(in_c, out_c, kernel_size=1, stride=1, padding=0)
        self.activate = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        y = self.res(x) + self.conv_short(x)
        y = self.activate(y)
        return y


class Conv3DBlock(nn.Module):
    """
    3D Convolution + Mish Activation
    """
    def __init__(self, in_c, out_c, ksize, stride):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=ksize,
            stride=stride,
            padding=ksize // 2
        )
        self.activation = nn.Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class RDBLOCK3D(nn.Module):
    """
    Residual Dense Block (3D Version)
    Structure is consistent with the original 2D RDBLOCK, only Conv2d -> Conv3d
    outchannel: Number of channels of input x (also the number of output channels of the first branch)
    """
    def __init__(self, outchannel):
        super(RDBLOCK3D, self).__init__()
        # x: (B, outchannel, D, H, W)
        self.conv1 = Conv3DBlock(in_c=outchannel,       out_c=outchannel,       ksize=1, stride=1)
        self.conv2 = Conv3DBlock(in_c=2 * outchannel,   out_c=outchannel,       ksize=3, stride=1)
        self.conv3 = Conv3DBlock(in_c=3 * outchannel,   out_c=outchannel,       ksize=1, stride=1)
        self.conv4 = Conv3DBlock(in_c=4 * outchannel,   out_c=2 * outchannel,   ksize=1, stride=1)
        self.shortcut = nn.Conv3d(in_channels=outchannel,
                                  out_channels=2 * outchannel,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        y = torch.cat((x, x1), dim=1)
        x2 = self.conv2(y)
        z = torch.cat((x, x1, x2), dim=1)
        x3 = self.conv3(z)
        x4 = self.conv4(torch.cat((z, x3), dim=1))
        out = self.shortcut(x) + x4
        out = F.mish(out)
        return out


# ===========================
# 3D Taylor Encoder
# ===========================

class TaylorEncoder3D(nn.Module):
    """
    3D Taylor Encoder:
    - base branch: semantic/basic structure features (output 1 channel)
    - gradient branch: recursively compute higher-order terms based on previous order output + raw input
    - final result = Σ (1 / i!) * y_i
    """
    def __init__(self, in_channels=1, n_taylor=2):
        super(TaylorEncoder3D, self).__init__()
        self.in_channels = in_channels
        self.n_taylor = n_taylor

        # base branch: C_in -> 32 -> 64 -> 32 -> 1
        self.base = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            ResB3D(32, 64),
            ResB3D(64, 32),
            ResB3D(32, 1),
        )

        # gradient branch:
        # Original version is 2 channels (y_prev + input), here we keep "1 + in_channels"
        grad_in_c = 1 + in_channels
        self.gradient = nn.Sequential(
            nn.Conv3d(grad_in_c, 8, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            RDBLOCK3D(8),   # output 16
            RDBLOCK3D(16),  # output 32
            RDBLOCK3D(32),  # output 64
            nn.Conv3d(64, 1, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        """
        x: (B, C_in, D, H, W)
        Returns:
          result: (B, 1, D, H, W)  — Taylor fusion result
          y_list: [y0, y1, ..., y_n]  1-channel features of each order
        """
        y_list = []
        base_feat = self.base(x)      # y0
        y_list.append(base_feat)

        # Initialize result
        result = torch.zeros_like(base_feat, device=x.device)

        # Recursively compute higher-order terms
        for i in range(1, self.n_taylor + 1):
            # cat: [y_{i-1}, input]
            grad_in = torch.cat([y_list[i - 1], x], dim=1)
            y_i = self.gradient(grad_in)
            y_list.append(y_i)

        # Taylor fusion: Σ (1 / i!) * y_i
        for i, y_i in enumerate(y_list):
            result = result + (1.0 / factorial(i)) * y_i

        return result, y_list


# ===========================
# TEM3D Main Network: Encoder + Segmentation Head
# ===========================

class TEM3D(nn.Module):
    """
    3D Taylor Encoder + Segmentation Head
    Interface style aligned with VNet:
      TEM3D(n_channels, n_classes, normalization, has_dropout, has_residual)
    Actually, normalization/has_residual are not used here, just to keep the interface.
    """
    def __init__(self,
                 n_channels=1,
                 n_classes=2,
                 n_filters=16,          # Placeholder parameter, to align with VNet interface
                 normalization='none',  # Placeholder
                 has_dropout=False,
                 has_residual=False,    # Placeholder
                 n_taylor=2):
        super(TEM3D, self).__init__()

        self.has_dropout = has_dropout

        # 3D Taylor Encoder
        self.encoder = TaylorEncoder3D(in_channels=n_channels, n_taylor=n_taylor)

        # Segmentation head: 1-channel Taylor fusion feature -> n_classes
        self.seg_head = nn.Conv3d(1, n_classes, kernel_size=1, stride=1, padding=0)

        if self.has_dropout:
            self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, x):
        """
        x: (B, n_channels, D, H, W)
        Returns:
          out_seg: (B, n_classes, D, H, W)
        If you want to use manifold features later, you can call self.encoder(x) separately to get result, y_list.
        """
        result, y_list = self.encoder(x)   # result: (B, 1, D, H, W)

        if self.has_dropout:
            result = self.dropout(result)

        out_seg = self.seg_head(result)    # (B, n_classes, D, H, W)
        return out_seg


if __name__ == "__main__":
    # Simple self-test
    x = torch.randn(1, 1, 32, 128, 128)  # (B, C, D, H, W)
    model = TEM3D(n_channels=1, n_classes=4, has_dropout=False, n_taylor=2)
    y = model(x)
    print("input:", x.shape)
    print("output:", y.shape)