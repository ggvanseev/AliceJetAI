qsub -l 'walltime=96:00:00' -q gpu-nv -V -o logfiles -j oe job_regular_training.sh
qsub -l 'walltime=96:00:00' -q gpu-nv -V -o logfiles -j oe job_regular_training.sh