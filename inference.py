import os
import glob
import monai
import torch
import numpy as np 
from tqdm import tqdm
import SimpleITK as sitk
from statistics import mean
from torch.optim import Adam
from natsort import natsorted
import matplotlib.pyplot as plt
from transformers import SamModel 
import matplotlib.patches as patches
from transformers import SamProcessor
from IPython.display import clear_output
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import threshold, normalize
from SamMedImg.dataloaders.SamDataLoader import SAMDataset, get_bounding_box
from SamDataLoader3ch import SAMDataset
from training import dice_score
# from configuration import SamConfig
# from transformers import (
#     SamVisionConfig,
#     SamPromptEncoderConfig,
#     SamMaskDecoderConfig,
#     SamModel,
# )

CKPT_PATH = '/raid/candi/Wen/SamMedImg/checkpoints/samseg'
# # Initializing a SamConfig with `"facebook/sam-vit-huge"` style configuration
# configuration = SamConfig()

# # Initializing a SamModel (with random weights) from the `"facebook/sam-vit-huge"` style configuration
# model = SamModel(configuration)
model = SamModel.from_pretrained("facebook/sam-vit-base")
base_dir = '.'
data_root = './datasets'
data_paths = {}
dataset = 'test'
data_types = ['2d_images_cat', '2d_masks']
# Create directories and print the number of images and masks in each

for data_type in data_types:
    # Construct the directory path
    dir_path = os.path.join(data_root, f'{dataset}_{data_type}')
        
    # Find images and labels in the directory
    if 'cat' in data_type:
        files = sorted(glob.glob(os.path.join(dir_path, "*t2w.nii.gz")))
    else:
        files = sorted(glob.glob(os.path.join(dir_path, "*.nii.gz")))
        
    # Store the image and label paths in the dictionary
    data_paths[f'{dataset}_{data_type.split("_")[1]}'] = files

print('Number of test images', len(data_paths['test_images']))

# create an instance of the processor for image preprocessing
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
print(processor)

# create test dataloader
test_dataset = SAMDataset(image_paths=data_paths['test_images'], mask_paths=data_paths['test_masks'], processor=processor)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#load model

ckpt = torch.load(CKPT_PATH+'/best_weights.pth')
model.load_state_dict(ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
i=0
# Iteratire through test images
with torch.no_grad():
    dice_vals = 0
    for batch in tqdm(test_dataloader):
        i +=1 
        # forward pass
        outputs = model(pixel_values=batch["pixel_values"].cuda(),
                      input_boxes=batch["input_boxes"].cuda(),
                      multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().cuda()
#         loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        

        # apply sigmoid
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        #pred  = torch.rand(1,256,256).float().cuda()
        dice_val = dice_score(medsam_seg_prob.round().squeeze(0), ground_truth_masks)
        dice_vals += dice_val
#        print(dice_val, outputs.pred_masks.min(), medsam_seg_prob.round().squeeze(0).max(), medsam_seg_prob.round().squeeze(0).min(), ground_truth_masks.max(), ground_truth_masks.min())
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        
        if i == len(test_dataloader):
            
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.imshow(batch["pixel_values"][0,1], cmap='gray')
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.imshow(batch["ground_truth_mask"][0], cmap='copper')
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.imshow(medsam_seg, cmap='copper')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('test.png')
    dice_mean = dice_vals/i
    print(dice_mean)
