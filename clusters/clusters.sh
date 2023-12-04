#!/bin/bash
#PBS -l select=1:ncpus=28
#PBS -q huge
#PBS -l walltime=336:00:00
#PBS -j oe
#PBS -o results.out
#PBS -N clusters
#PBS -m bea

cd $PBS_O_WORKDIR

#load anaconda
module load anaconda3/2022.05

source activate pdbclust
for f in */
do
  mkdir "$f"/results
  python main.py input="$f" output="$f"/results cpu_threads=28 noh=true method=dbscan eps=1.9 min_samples=1
done

