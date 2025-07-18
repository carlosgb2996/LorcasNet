# -*- coding: utf-8 -*-

import argparse
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np
import models                    # exporta LorcasNet y LorcasNet_bn
from multiscaleloss import multiscaleEPE, realEPE
from util import AverageMeter, save_checkpoint

# Configuración del dispositivo (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Modelos disponibles
model_names = ['LorcasNet', 'LorcasNet_bn']

parser = argparse.ArgumentParser(
    description='Entrenamiento de LorcasNet en un conjunto de datos de moteado',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--arch', default='LorcasNet_bn',
                    choices=model_names,
                    help='Selecciona LorcasNet (sin BN) o LorcasNet_bn (con BatchNorm)')
parser.add_argument('--solver', default='adamw', choices=['adamw', 'sgd'],
                    help='Optimizador')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='Num. workers DataLoader')
parser.add_argument('--epochs', default=100, type=int,
                    help='Épocas totales')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='Tamaño de batch')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    help='Tasa de aprendizaje inicial')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='Weight decay para pesos')
parser.add_argument('--bias-decay', default=0, type=float,
                    help='Weight decay para sesgos')
parser.add_argument('--multiscale-weights', '-w', nargs=5, type=float,
                    default=[0.005, 0.01, 0.02, 0.08, 0.32],
                    help='Pesos multiescala (baja→alta resolución)')
parser.add_argument('--sparse', action='store_true',
                    help='Usar EPE escaso (considera NaN)')
parser.add_argument('--lambda-conf', default=1.0, type=float,
                    help='Peso de la pérdida de confianza')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    help='Frecuencia de impresión')
parser.add_argument('--save-path', default='./checkpoints',
                    help='Directorio para checkpoints')

best_EPE = float('inf')

class SpecklesDataset(Dataset):
    """
    Dataset de moteado que devuelve:
      - 'input': tensor [2,H,W] con Ref y Def concatenadas
      - 'Dispx', 'Dispy': cada uno [1,H,W]
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        ref_path   = os.path.join(self.root_dir, row[0])
        def_path   = os.path.join(self.root_dir, row[1])
        dispx_path = os.path.join(self.root_dir, row[2])
        dispy_path = os.path.join(self.root_dir, row[3])

        Ref   = np.genfromtxt(ref_path,   delimiter=',')
        Def   = np.genfromtxt(def_path,   delimiter=',')
        Dispx = np.genfromtxt(dispx_path, delimiter=',')
        Dispy = np.genfromtxt(dispy_path, delimiter=',')

        # a torch tensor y normalización
        Ref   = torch.from_numpy(Ref).unsqueeze(0).float()
        Def   = torch.from_numpy(Def).unsqueeze(0).float()
        Dispx = torch.from_numpy(Dispx).unsqueeze(0).float()
        Dispy = torch.from_numpy(Dispy).unsqueeze(0).float()

        # input de 2 canales
        input = torch.cat([Ref, Def], dim=0)  # [2, H, W]

        sample = {'input': input, 'Dispx': Dispx, 'Dispy': Dispy}
        if self.transform:
            sample = self.transform(sample)
        return sample
class Normalization(object):
    """
    Normaliza las imágenes y los desplazamientos antes de pasarlos al modelo.

    - Las imágenes (2 canales: Ref y Def) se normalizan dividiendo por 255.0.
    - Los desplazamientos normalizados de [-1,1] a [0,1].
    """
    def __call__(self, sample):
        # 'input': tensor [2,H,W]; 'Dispx','Dispy': [1,H,W]
        inp = sample['input']       # tensor float
        dx  = sample['Dispx']
        dy  = sample['Dispy']

        # Normalizar imágenes a [0,1]
        inp = inp / 255.0

        # Normalizar desplazamientos de [-1,1] a [0,1]
        dx = (dx + 1.0) * 0.5
        dy = (dy + 1.0) * 0.5

        return {'input': inp, 'Dispx': dx, 'Dispy': dy}


def main():
    args = parser.parse_args()

    # Directorio de checkpoints con metadatos
    save_path = os.path.join(
        args.save_path,
        f"{args.arch},{args.solver},{args.epochs}ep,b{args.batch_size},lr{args.lr}"
    )
    os.makedirs(save_path, exist_ok=True)

    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    val_writer   = SummaryWriter(os.path.join(save_path, 'val'))

    # Load datasets
    transform = Normalization()
    train_set = SpecklesDataset(
        csv_file='Dataset/Train_annotations.csv',
        root_dir='Dataset/Train_Data',
        transform=transform
    )
    val_set = SpecklesDataset(
        csv_file='Dataset/Test_annotations.csv',
        root_dir='Dataset/Test_Data',
        transform=transform
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True
    )

    print(f"{len(train_set)+len(val_set)} muestras — {len(train_set)} train / {len(val_set)} val")

    # Modelo
    model = models.__dict__[args.arch]().to(device)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    # Optimizador y scheduler
    param_groups = [
        {'params': model.module.bias_parameters(),  'weight_decay': args.bias_decay},
        {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay}
    ]
    optimizer = (
        optim.AdamW(param_groups, lr=args.lr)
        if args.solver == 'adamw'
        else optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = GradScaler()
     # Early stopping
    early_stopping_patience = 30
    no_improve_epochs = 0
    for epoch in range(args.epochs):
        train_loss, train_epe = train_one_epoch(train_loader, model,
                                                optimizer, scaler, epoch, args)
        train_writer.add_scalar('EPE', train_epe, epoch)

        val_epe = validate(val_loader, model, epoch, args)
        val_writer.add_scalar('EPE', val_epe, epoch)

        is_best = val_epe < best_EPE
        best_EPE = min(val_epe, best_EPE)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_epe': best_EPE,
            'optimizer': optimizer.state_dict()
        }, is_best, save_path)

        scheduler.step()
        # Early stopping check
        if not is_best:
            no_improve_epochs += 1
        else:
            no_improve_epochs = 0

        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping tras {epoch+1} épocas sin mejora.")
            break

def train_one_epoch(train_loader, model, optimizer, scaler, epoch, args):
    """
    Entrena el modelo una época usando mixed precision,
    pérdida multiescala + confianza heteroscedástica.
    """
    model.train()
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    meters     = {k: AverageMeter() for k in ['loss', 'epe', 'conf_loss']}
    end = time.time()

    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        inp    = batch['input'].to(device)                              # [B,2,H,W]
        target = torch.cat([batch['Dispx'], batch['Dispy']], dim=1).to(device)  # [B,2,H,W]

        optimizer.zero_grad()
        with autocast():
            flows, conf = model(inp)[:-1], model(inp)[-1]  # flows: list of flow2…flow6, conf: [B,1,H,W]

            # Pérdida multiescala
            loss_ms = multiscaleEPE(flows, target,
                                    weights=args.multiscale_weights,
                                    sparse=args.sparse)

            # Pérdida de confianza (heteroscedástica)
            sq_err    = (flows[0] - target).pow(2)       # [B,2,H,W]
            conf_map  = conf.repeat(1, 2, 1, 1)          # [B,2,H,W]
            loss_conf = (conf_map * sq_err - torch.log(conf_map + 1e-6)).mean()

            loss = loss_ms + args.lambda_conf * loss_conf

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Métricas
        epe = realEPE(flows[0], target, sparse=args.sparse)
        meters['loss'].update(loss.item(), inp.size(0))
        meters['epe'].update(epe.item(), inp.size(0))
        meters['conf_loss'].update(loss_conf.item(), inp.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f"Epoch[{epoch}][{i}/{len(train_loader)}] "
                  f"Time {batch_time.val:.3f}s  Data {data_time.val:.3f}s  "
                  f"Loss {meters['loss'].val:.4f}  EPE {meters['epe'].val:.4f}  "
                  f"ConfLoss {meters['conf_loss'].val:.4f}")

    return meters['loss'].avg, meters['epe'].avg


def validate(val_loader, model, epoch, args):
    """
    Evalúa el modelo en el conjunto de validación, 
    reportando el EPE medio de la escala más fina.
    """
    model.eval()
    meter = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inp    = batch['input'].to(device)
            target = torch.cat([batch['Dispx'], batch['Dispy']], dim=1).to(device)

            flow2, _ = model(inp)[:2]  # flow2 es la predicción más fina
            epe = realEPE(flow2, target, sparse=args.sparse)
            meter.update(epe.item(), inp.size(0))

            if i % args.print_freq == 0:
                print(f"Val[{i}/{len(val_loader)}] "
                      f"Time {(time.time()-end):.3f}s  EPE {meter.val:.4f}")
            end = time.time()

    print(f" * Validation EPE: {meter.avg:.4f}")
    return meter.avg

if __name__ == '__main__':
    main()