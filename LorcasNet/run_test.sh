#!/usr/bin/env bash
#PBS -P w09                              
#PBS -q gpuvolta                         
#PBS -N lorcasnet_train_test             
#PBS -l ncpus=12                        
#PBS -l ngpus=1                         
#PBS -l mem=16GB                        
#PBS -l walltime=00:10:00  
#PBS -o debug_${PBS_JOBID}.out
#PBS -e debug_${PBS_JOBID}.err

set -euo pipefail

# DEBUG: identificar nodo y montajes para w09
echo "=== HOSTNAME: $(hostname) ==="
echo "=== MOUNTS for w09 ==="
mount | grep w09 || echo "No mounts for w09"

echo "=== Checking /g/data/w09/cg2265/LorcasNet ==="
ls -ld /g/data/w09/cg2265/LorcasNet || echo "Not found"

echo "=== Checking /g/data/w09/cg2265/LorcasNet/LorcasNet ==="
ls -ld /g/data/w09/cg2265/LorcasNet/LorcasNet || echo "Not found"

echo "=== Checking /scratch/w09/cg2265/LorcasNet ==="
ls -ld /scratch/w09/cg2265/LorcasNet || echo "Not found"

echo "=== Checking /scratch/w09/cg2265/LorcasNet/LorcasNet ==="
ls -ld /scratch/w09/cg2265/LorcasNet/LorcasNet || echo "Not found"

echo "=== Debug finished ==="
