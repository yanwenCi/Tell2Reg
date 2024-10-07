#$ -S /bin/bash
#$ -j y
#$ -N sam-as
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=5:00:00
#$ -l gpu=true
#$ -wd /cluster/project7/longitude/SamMedImg

conda activate ./env-sam
python training_with_text_3dAS.py --data_root ../Datasets/AS-morph-interp-ver/0.7-0.7-0.7-64-64-51/
