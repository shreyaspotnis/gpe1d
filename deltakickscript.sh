#!/bin/bash
# Torque submission script for SciNet ARC
#
#PBS -l nodes=2:ppn=8:gpus=2,walltime=4:00:00
#PBS -N GPUtest
#PBS -q arc
cd $PBS_O_WORKDIR
 
# EXECUTION COMMAND; -np = nodes*ppn
 cd $HOME/gpe1d
 module load cuda/4.0
 python $HOME/gpe1d/deltakickjob.py >& $SCRATCH/deltakicklog
