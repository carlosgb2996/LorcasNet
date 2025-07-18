#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import models  # exporta LorcasNet y LorcasNet_bn

def parse_args():
    parser = argparse.ArgumentParser(description='Inferencia con LorcasNet')
    parser.add_argument('--arch',        choices=['LorcasNet','LorcasNet_bn'],
                        default='LorcasNet_bn', help='Arquitectura a usar')
    parser.add_argument('--model-path',  required=True, help='Checkpoint .pth')
    parser.add_argument('--img1',        required=True, help='CSV de imagen Ref')
    parser.add_argument('--img2',        required=True, help='CSV de imagen Def')
    parser.add_argument('--save-dir',    default='output', help='Dónde guardar resultados')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carga del modelo
    model = getattr(models, args.arch)().to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Carga y normalización de imágenes
    img1 = np.loadtxt(args.img1, delimiter=',').astype(np.float32) / 255.0
    img2 = np.loadtxt(args.img2, delimiter=',').astype(np.float32) / 255.0

    # Asegurar shape [1,H,W]
    if img1.ndim == 2:
        img1 = img1[None,:,:]
        img2 = img2[None,:,:]

    # Tensor de entrada [1,2,H,W]
    inp = np.stack([img1, img2], axis=0)[None,...]
    inp = torch.from_numpy(inp).to(device)

    # Inferencia
    with torch.no_grad():
        outputs = model(inp)
    # outputs = (flow2, flow3, flow4, flow5, flow6, conf2)
    flow2, conf2 = outputs[0], outputs[-1]

    # Upsample al tamaño original
    H, W = img1.shape[1:]
    flow2 = F.interpolate(flow2, size=(H,W), mode='bilinear', align_corners=False)
    conf2 = F.interpolate(conf2, size=(H,W), mode='bilinear', align_corners=False)

    # Desnormalizar flujo de [0,1] a [-1,1]
    flow_np = flow2[0].cpu().numpy()  # [2,H,W]
    disp_x = flow_np[0] * 2.0 - 1.0
    disp_y = flow_np[1] * 2.0 - 1.0

    # Convertir confianza a numpy
    conf_np = conf2[0,0].cpu().numpy()  # [H,W]

    # Guardar resultados
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.img1).stem.replace('Ref','Def')
    np.savetxt(save_dir / f"{base}_disp_x.csv", disp_x, delimiter=',')
    np.savetxt(save_dir / f"{base}_disp_y.csv", disp_y, delimiter=',')
    np.savetxt(save_dir / f"{base}_conf.csv",   conf_np, delimiter=',')

    print("Inferencia completa. Archivos guardados en", save_dir)

if __name__ == '__main__':
    main()
