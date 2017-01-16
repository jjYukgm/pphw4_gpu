# NOTICE: Please do not remove the '#' before 'PBS'

# Name of your job
#PBS -N CUDA_JOB

# Declaring job as not re-runnable
#PBS -r n

# Resource allocation (how many nodes? how many gpus per node?)
#PBS -l nodes=1:ppn=1:gpus=1:exclusive_process

# Max execution time of your job (hh:mm:ss)
# Your job may got killed if you exceed this limit
#PBS -l walltime=00:05:00
#PBS -e ecq1
#PBS -o ocq1

cd $PBS_O_WORKDIR

# Execute cuda version
time ./HW4_105061516_cuda.exe testcase/in5 out 64
