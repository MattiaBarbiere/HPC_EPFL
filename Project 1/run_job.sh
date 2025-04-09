#!/bin/bash
#SBATCH --qos=math-454
#SBATCH --account=math-454
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --output=SS_%x_nt${NTASKS}.txt
#SBATCH --job-name=cg_${NTASKS}

echo "Running with ${NTASKS} tasks"

srun --ntasks=${NTASKS} ./cgsolver lap2D_5pt_n100.mtx
srun --ntasks=${NTASKS} ./cgsolver lap2D_5pt_n200.mtx
srun --ntasks=${NTASKS} ./cgsolver lap2D_5pt_n300.mtx
srun --ntasks=${NTASKS} ./cgsolver lap2D_5pt_n600.mtx
srun --ntasks=${NTASKS} ./cgsolver lap2D_5pt_n1000.mtx
srun --ntasks=${NTASKS} ./cgsolver lap2D_5pt_n2000.mtx