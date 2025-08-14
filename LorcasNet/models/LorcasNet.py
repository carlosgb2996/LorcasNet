import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_
from .util import conv as conv_block, predict_confidence, crop_like, ResidualBlock, CBAMBlock

__all__ = ["LorcasNet", "LorcasNet_bn"]

# ──────────────────────────────────────────────────────────────────────────────
# Flow head LINEAL (sin tanh) → predice (u,v) en PIXELES
# ──────────────────────────────────────────────────────────────────────────────
def predict_flow_linear(c_in: int) -> nn.Module:
    return nn.Conv2d(c_in, 2, kernel_size=3, stride=1, padding=1)

class LorcasNet(nn.Module):
    """
    Cambios clave:
      - Cabeza de flujo lineal (pixeles).
      - Inicialización configurable de cabezas de flujo:
          * zero_flow_heads_init=True  → TODAS las cabezas de flujo a CERO.
          * zero_flow_heads_init=False → Kaiming normal (despegue más rápido).
      - Capa de calibración aprendible por canal al final:
          flow2_cal = flow2 * out_scale + out_bias
        * out_scale es (1,2,1,1). Por defecto ≈0.2 para acelerar convergencia
          hacia rangos pequeños observados; puedes cambiarlo con out_scale_init.
        * Se puede desactivar/“congelar” con learn_output_calib=False.
      - Confianza forzada a ser positiva en forward() con softplus + 1e-6.
      - Helpers set_output_calibration(...) / freeze_output_calibration(...)
        para inyectar factores del check-scale (A/B0) en inferencia si hace falta.
    """
    expansion = 1
    def __init__(
        self,
        batchNorm: bool = True,
        dropout_prob: float = 0.1,
        # Calibración de salida
        learn_output_calib: bool = True,
        out_scale_init: float = 0.20,   # ~rango observado (B0 ≈ 0.08–0.16)
        out_bias_init: float = 0.0,
        calibrate_all_scales: bool = False,  # si True: aplica a flow3..flow6 (normalmente False)
        # NUEVO: control de init en las cabezas
        zero_flow_heads_init: bool = True,
    ):
        super().__init__()
        self.batchNorm = batchNorm
        self.dropout_prob = dropout_prob
        self.calibrate_all_scales = calibrate_all_scales

        # Encoder
        self.conv1 = conv_block(self.batchNorm, 2,   64, kernel_size=7, stride=1)
        self.res1  = ResidualBlock(64,  64,  batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)

        self.conv2 = conv_block(self.batchNorm, 64,  128, kernel_size=5, stride=1)
        self.res2  = ResidualBlock(128, 128, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)

        self.conv3 = conv_block(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.res3  = ResidualBlock(256, 256, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)

        self.conv4 = conv_block(self.batchNorm, 256, 512, stride=2)
        self.res4  = ResidualBlock(512, 512, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)

        self.conv5 = conv_block(self.batchNorm, 512, 512, stride=2)
        self.res5  = ResidualBlock(512, 512, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)

        self.conv6 = conv_block(self.batchNorm, 512, 1024, stride=2)
        self.res6  = ResidualBlock(1024, 1024, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)

        # Decoder
        self.deconv5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res_deconv5 = ResidualBlock(512, 512, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)
        self.cbam5 = CBAMBlock(512)

        self.deconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(1026, 256, 3, padding=1),  # 512(out5)+512(d5)+2(flow6_up)=1026
            nn.ReLU(inplace=True),
        )
        self.res_deconv4 = ResidualBlock(256, 256, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)
        self.cbam4 = CBAMBlock(256)

        self.deconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(770, 128, 3, padding=1),  # 512(out4)+256(d4)+2(flow5_up)=770
            nn.ReLU(inplace=True),
        )
        self.res_deconv3 = ResidualBlock(128, 128, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)
        self.cbam3 = CBAMBlock(128)

        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(386, 64, 3, padding=1),   # 256(out3)+128(d3)+2(flow4_up)=386
            nn.ReLU(inplace=True),
        )
        self.res_deconv2 = ResidualBlock(64, 64, batchNorm=self.batchNorm, dropout_prob=self.dropout_prob)
        self.cbam2 = CBAMBlock(64)

        # Flow & confidence heads (linear → pixels)
        self.predict_flow6       = predict_flow_linear(1024)
        self.predict_flow5       = predict_flow_linear(1026)
        self.predict_flow4       = predict_flow_linear(770)
        self.predict_flow3       = predict_flow_linear(386)
        self.predict_flow2       = predict_flow_linear(194)   # 128(out2)+64(d2)+2(flow3_up)=194
        self.predict_confidence2 = predict_confidence(194)

        # Upsampling layers
        self.upsampled_flow6_to_5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsampled_flow5_to_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsampled_flow4_to_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsampled_flow3_to_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # ── Capa de calibración de salida (por canal) ──────────────────────────
        # Parámetros shape (1,2,1,1) para broadcast sobre (B,2,H,W)
        self.out_scale = nn.Parameter(
            torch.tensor([out_scale_init, out_scale_init], dtype=torch.float32).view(1, 2, 1, 1),
            requires_grad=learn_output_calib
        )
        self.out_bias  = nn.Parameter(
            torch.tensor([out_bias_init, out_bias_init], dtype=torch.float32).view(1, 2, 1, 1),
            requires_grad=learn_output_calib
        )

        # Init estándar
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

        # Inicialización de las cabezas de flujo
        flow_heads = [self.predict_flow6, self.predict_flow5, self.predict_flow4,
                      self.predict_flow3, self.predict_flow2]
        if zero_flow_heads_init:
            # Arranque sin sesgo: pesos y bias a 0
            for head in flow_heads:
                conv = head[0] if isinstance(head, nn.Sequential) else head
                nn.init.zeros_(conv.weight)
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)
        # Si zero_flow_heads_init=False, ya quedaron con Kaiming (arranque con más “empuje”)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers públicos para calibración en inferencia
    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def set_output_calibration(self, scale_xy=(1.0, 1.0), bias_xy=(0.0, 0.0), requires_grad: bool = False):
        """Fija (opcionalmente congela) la calibración de salida desde coef. externos (A/B0)."""
        sx, sy = float(scale_xy[0]), float(scale_xy[1])
        bx, by = float(bias_xy[0]),  float(bias_xy[1])
        self.out_scale.copy_(torch.tensor([[[[sx]], [[sy]]]], dtype=self.out_scale.dtype, device=self.out_scale.device))
        self.out_bias.copy_( torch.tensor([[[[bx]], [[by]]]], dtype=self.out_bias.dtype, device=self.out_bias.device))
        self.out_scale.requires_grad_(requires_grad)
        self.out_bias.requires_grad_(requires_grad)

    @torch.no_grad()
    def freeze_output_calibration(self):
        """Congela la capa de salida (útil si set_output_calibration se fija desde check-scale)."""
        self.out_scale.requires_grad_(False)
        self.out_bias.requires_grad_(False)

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, x):
        # Encoder
        out1 = self.res1(self.conv1(x))
        out2 = self.res2(self.conv2(out1))
        out3 = self.res3(self.conv3(out2))
        out4 = self.res4(self.conv4(out3))
        out5 = self.res5(self.conv5(out4))
        out6 = self.res6(self.conv6(out5))

        # Coarse
        flow6    = self.predict_flow6(out6)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out5)
        d5       = crop_like(self.deconv5(out6), out5)
        d5       = self.cbam5(self.res_deconv5(d5))

        # 5
        cat5   = torch.cat((out5, d5, flow6_up), 1)
        flow5  = self.predict_flow5(cat5)
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out4)
        d4     = crop_like(self.deconv4(cat5), out4)
        d4     = self.cbam4(self.res_deconv4(d4))

        # 4
        cat4   = torch.cat((out4, d4, flow5_up), 1)
        flow4  = self.predict_flow4(cat4)
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out3)
        d3     = crop_like(self.deconv3(cat4), out3)
        d3     = self.cbam3(self.res_deconv3(d3))

        # 3
        cat3   = torch.cat((out3, d3, flow4_up), 1)
        flow3  = self.predict_flow3(cat3)
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out2)
        d2     = crop_like(self.deconv2(cat3), out2)
        d2     = self.cbam2(self.res_deconv2(d2))

        # 2
        cat2  = torch.cat((out2, d2, flow3_up), 1)
        flow2 = self.predict_flow2(cat2)          # ← pixels (sin tanh)
        conf2 = self.predict_confidence2(cat2)
        # NUEVO: confianza positiva y estable numéricamente
        conf2 = F.softplus(conf2) + 1e-6

        # Calibración final
        def _cal(f):
            return f * self.out_scale + self.out_bias

        if self.calibrate_all_scales:
            flow6 = _cal(flow6); flow5 = _cal(flow5); flow4 = _cal(flow4); flow3 = _cal(flow3)

        flow2_cal = _cal(flow2)

        # Mantén compatibilidad con tu pipeline:
        #  - training: tu código espera multi-escala + conf como tupla
        #  - eval: muchos scripts aceptan dict {'flow','conf'}
        if self.training:
            return (flow2_cal,
                    flow3 if not self.calibrate_all_scales else _cal(flow3),
                    flow4 if not self.calibrate_all_scales else _cal(flow4),
                    flow5 if not self.calibrate_all_scales else _cal(flow5),
                    flow6 if not self.calibrate_all_scales else _cal(flow6),
                    conf2)
        else:
            return {"flow": flow2_cal, "conf": conf2}

    # Param groups
    def weight_parameters(self):
        return [p for n, p in self.named_parameters() if "weight" in n]

    def bias_parameters(self):
        return [p for n, p in self.named_parameters() if "bias" in n]

_LorcasNet = LorcasNet

def LorcasNet(data=None):
    m = _LorcasNet(batchNorm=False)
    (data and m.load_state_dict(data["state_dict"]))
    return m

def LorcasNet_bn(data=None):
    m = _LorcasNet(batchNorm=True)
    (data and m.load_state_dict(data["state_dict"]))
    return m
