# Loop over desired task counts
for ntasks in 1 2 4 6 8; do
    sbatch --ntasks=$ntasks --export=NTASKS=$ntasks run_job.sh
done