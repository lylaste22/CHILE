#!/bin/bash

## Job Name
#SBATCH -J myscript

## Job Output (stdout + stderr) file
## %A = parent job id
## %a = task id
## %j = real job id
#SBATCH -o myscript.%A.%a-%j.out

# Processors:
# Number of MPI tasks
#SBATCH -n 1
#
# Number of cores per task
#SBATCH -c 10

# Queue name (partition name)
#SBATCH --partition=slims

## Job Array tasks (from 1 to 10)
#SBATCH --array=1-4

## Email notification (all events)
##SBATCH -M jcm@dim.uchile.cl
##SBATCH --mail-type=all

# parametrize your variables
PYTHON=python

# get access to the SLURM task id variable 
export SLURM_JOBID               ; # real job id
export SLURM_ARRAY_JOB_ID        ; # parent job id (originator job)
export SLURM_ARRAY_TASK_ID       ; # task id

if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
        # not under slurm environment.
        # assuming only one input file
        SLURM_ARRAY_TASK_ID=1
fi

# make the toolchain available
module load astro

# list the modules loaded
module list

# include your home bin directory into the PATH
export PATH=$HOME/bin:$PATH

# export the OpenMP number of thread requested
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [[ -z "$OMP_NUM_THREADS" ]]; then
     # no openmp requested to slurm. setting to default
     OMP_NUM_THREADS=1
fi

echo "Number of thread per task : $OMP_NUM_THREADS"

# write the hostname where we are running
echo "Hostname: `hostname`"

## My Script ##

export PYTHONPATH=`pwd`/test_omp:$PYTHONPATH

${PYTHON} myscript.py arg1 arg2 arg3
EXIT_CODE=$?

## finishing tasks

echo "Exit code: $EXIT_CODE" 

##  - remove temporary files
##  - handle errors

