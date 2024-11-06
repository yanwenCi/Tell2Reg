# Tell2Reg
## Introduction
This paper proposes a novel registration method that leverages object-based region correspondences (ROIs) to achieve image registration. The proposed approach is training-free, utilizing pre-trained models such as GroundingDINO and SAM to generate corresponding ROIs, enabling a text-ROI-image workflow for registration. 


<img src="net.png" width="400"/>


## Installation 
conda create env-tell
pip install -r requirements.txt

## Usage
training_with_text3d.py --dataroot datasets --gpu 0



