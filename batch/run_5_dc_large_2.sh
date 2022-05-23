#!/bin/bash
#SBATCH --export=ALL # export all environment variables to the batch job
#SBATCH -D . # set working directory to .
#SBATCH -p highmem # submit to the sequential queue
#SBATCH --time=168:00:00 # maximum walltime for the job
#SBATCH -A Research_Project-T101197 # research project to submit under
#SBATCH --nodes=1 # specify number of nodes
#SBATCH --ntasks-per-node=16 # specify number of processors per node
#SBATCH --mail-type=END # send email at job completion
#SBATCH --mail-user=jb931@exeter.ac.uk # email address
#SBATCH --mem=128G

cd ../NFV_AG_Journal/src/alg
cargo run --release -- -i=5 -s=2 -z=3 -t=DCell
