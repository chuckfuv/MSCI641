#!/bin/bash
#SBATCH --account=def-aghuang
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=3:00:00
#SBATCH --mail-user=fuv@me.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python ass2.py pos_train.csv neg_train.csv  pos_val.csv  neg_val.csv pos_test.csv neg_test.csv
python ass2.py pos_train_no_stopword.csv neg_train_no_stopword.csv  pos_val_no_stopword.csv  neg_val_no_stopword.csv pos_test_no_stopword.csv neg_test_no_stopword.csv