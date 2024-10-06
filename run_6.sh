#$ -S /bin/bash
#$ -j y
#$ -N sam-t
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=40:00:00
#$ -l gpu=true
#$ -wd ./logs

conda activate ./env-sam
python training_with_text_3dimg.py --data_root datasets/abdomen_data
