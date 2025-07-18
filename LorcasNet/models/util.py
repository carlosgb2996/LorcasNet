import torch
import torch.nn as nn

# Convolutional block: Conv2d -> (BatchNorm) -> (LeakyReLU)
def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, activate=True):
    """
    Crea una capa Conv2d opcionalmente seguida de BatchNorm y LeakyReLU.
    """
    padding = kernel_size // 2
    if batchNorm:
        layers = [
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_planes)
        ]
        if activate:
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        return nn.Sequential(*layers)
    else:
        conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=True)
        if activate:
            return nn.Sequential(conv_layer, nn.LeakyReLU(0.1, inplace=True))
        else:
            return conv_layer

class ResidualBlock(nn.Module):
    """
    Bloque Residual con dos convoluciones y conexión de salto.
    """
    def __init__(self, in_planes, out_planes, stride=1, batchNorm=True, dropout_prob=0.1):
        super(ResidualBlock, self).__init__()
        self.batchNorm = batchNorm
        self.dropout_prob = dropout_prob
        self.conv1 = conv(self.batchNorm, in_planes, out_planes, kernel_size=3, stride=stride)
        self.conv2 = conv(self.batchNorm, out_planes, out_planes, kernel_size=3, stride=1, activate=False)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout(p=self.dropout_prob) if self.dropout_prob > 0 else nn.Identity()

        if stride != 1 or in_planes != out_planes:
            layers = [
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            ]
            if self.batchNorm:
                layers.append(nn.BatchNorm2d(out_planes))
            self.shortcut = nn.Sequential(*layers)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Predicción de flujo de desplazamiento: 2 canales
def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)

# Predicción de confianza: 1 canal + Sigmoid
def predict_confidence(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Sigmoid()
    )

# Deconvolución (no usada actualmente)
def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True)
    )

# Recorta `input` al tamaño de `target`
def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]

class CBAMBlock(nn.Module):
    """
    Módulo de atención de canal & espacial (CBAM).
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        # Atención de canal
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        # Atención espacial
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                      padding=kernel_size//2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Canal
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg_out = self.mlp(avg)
        max_out = self.mlp(mx)
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_att

        # Espacial
        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_sp, max_sp], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial))
        x = x * spatial_att
        return x
