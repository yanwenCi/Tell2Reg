#$ -S /bin/bash
#$ -j y
#$ -N sam-t
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=5:00:00
#$ -l gpu=true
#$ -wd /cluster/project7/longitude/SamMedImg

conda activate ./env-sam
python training_with_text_3dimg.py --data_root datasets/abdomen_data
