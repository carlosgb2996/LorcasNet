import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .util import conv, predict_flow, predict_confidence, crop_like, ResidualBlock, CBAMBlock

__all__ = [
    'LorcasNet', 'LorcasNet_bn'
]

class LorcasNet(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True, dropout_prob=0.1):
        super(LorcasNet, self).__init__()
        self.batchNorm = batchNorm
        self.dropout_prob = dropout_prob

        # Encoder
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=1)
        self.res1  = ResidualBlock(64, 64, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)

        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=1)
        self.res2  = ResidualBlock(128, 128, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)

        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.res3  = ResidualBlock(256, 256, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)

        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.res4  = ResidualBlock(512, 512, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)

        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.res5  = ResidualBlock(512, 512, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)

        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.res6  = ResidualBlock(1024, 1024, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)

        # Decoder with residual & CBAM
        self.deconv5     = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res_deconv5 = ResidualBlock(512, 512, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)
        self.cbam5       = CBAMBlock(512)

        self.deconv4     = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1026, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res_deconv4 = ResidualBlock(256, 256, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)
        self.cbam4       = CBAMBlock(256)

        self.deconv3     = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(770, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res_deconv3 = ResidualBlock(128, 128, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)
        self.cbam3       = CBAMBlock(128)

        self.deconv2     = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(386, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res_deconv2 = ResidualBlock(64, 64, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)
        self.cbam2       = CBAMBlock(64)

        # Flow & confidence prediction
        self.predict_flow6       = predict_flow(1024)
        self.predict_flow5       = predict_flow(1026)
        self.predict_flow4       = predict_flow(770)
        self.predict_flow3       = predict_flow(386)
        self.predict_flow2       = predict_flow(194)
        self.predict_confidence2 = predict_confidence(194)

        # Upsampling layers for flow
        self.upsampled_flow6_to_5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsampled_flow5_to_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsampled_flow4_to_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsampled_flow3_to_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        out_conv1 = self.conv1(x)
        out_res1  = self.res1(out_conv1)

        out_conv2 = self.conv2(out_res1)
        out_res2  = self.res2(out_conv2)

        out_conv3 = self.conv3(out_res2)
        out_res3  = self.res3(out_conv3)

        out_conv4 = self.conv4(out_res3)
        out_res4  = self.res4(out_conv4)

        out_conv5 = self.conv5(out_res4)
        out_res5  = self.res5(out_conv5)

        out_conv6 = self.conv6(out_res5)
        out_res6  = self.res6(out_conv6)

        # Coarse flow
        flow6      = self.predict_flow6(out_res6)
        flow6_up   = crop_like(self.upsampled_flow6_to_5(flow6), out_res5)
        out_d5     = crop_like(self.deconv5(out_res6), out_res5)
        out_d5     = self.res_deconv5(out_d5)
        out_d5     = self.cbam5(out_d5)

        # Flow5
        concat5    = torch.cat((out_res5, out_d5, flow6_up), 1)
        flow5      = self.predict_flow5(concat5)
        flow5_up   = crop_like(self.upsampled_flow5_to_4(flow5), out_res4)
        out_d4     = crop_like(self.deconv4(concat5), out_res4)
        out_d4     = self.res_deconv4(out_d4)
        out_d4     = self.cbam4(out_d4)

        # Flow4
        concat4    = torch.cat((out_res4, out_d4, flow5_up), 1)
        flow4      = self.predict_flow4(concat4)
        flow4_up   = crop_like(self.upsampled_flow4_to_3(flow4), out_res3)
        out_d3     = crop_like(self.deconv3(concat4), out_res3)
        out_d3     = self.res_deconv3(out_d3)
        out_d3     = self.cbam3(out_d3)

        # Flow3
        concat3    = torch.cat((out_res3, out_d3, flow4_up), 1)
        flow3      = self.predict_flow3(concat3)
        flow3_up   = crop_like(self.upsampled_flow3_to_2(flow3), out_res2)
        out_d2     = crop_like(self.deconv2(concat3), out_res2)
        out_d2     = self.res_deconv2(out_d2)
        out_d2     = self.cbam2(out_d2)

        # Flow2 & confidence
        concat2    = torch.cat((out_res2, out_d2, flow3_up), 1)
        flow2      = self.predict_flow2(concat2)
        conf2      = self.predict_confidence2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6, conf2
        else:
            return flow2, conf2

    def weight_parameters(self):
        return [p for n, p in self.named_parameters() if 'weight' in n]

    def bias_parameters(self):
        return [p for n, p in self.named_parameters() if 'bias' in n]


def LorcasNet(data=None):
    model = LorcasNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def LorcasNet_bn(data=None):
    model = LorcasNet(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
