import os
import glob
import monai
import torch
from torch import nn
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
import os.path as osp
from SamMedImg.dataloaders.SamDataLoader import SAMDataset, get_bounding_box
from SamMedImg.dataloaders.LangDataLoader import SAMDataset
from networks.networks import SamWithTextPrompt, draw_image
# from networks.networks_with_pretrain import SamWithTextPrompt
from text_prompts import generate_prompts

#matplotlib inline



def dice_score(prediction, target):
    smooth = 1e-10 # Smoothing factor to prevent division by zero
    #print(prediction.shape, target.shape)
    prediction = torch.sigmoid(prediction)
    batch_size = prediction.size(0)
    prediction_flat = prediction.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)

    intersection = torch.sum(prediction_flat * target_flat, dim=-1)
    union = torch.sum(prediction_flat, dim=-1) + torch.sum(target_flat, dim=-1)
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.mean().item()  # Return the dice score as a scalar value

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

# Create directories and print the number of images and masks in each
def creat_datasets(datasets, data_root, data_types):
    # Initialize dictionary for storing image and label paths
    data_paths = {}
    for dataset in datasets:
        for data_type in data_types:
            # Construct the directory path
            dir_path = os.path.join(data_root, f'{dataset}_{data_type}')
            
            # Find images and labels in the directory
            if 'cat' in data_type:
                files = sorted(glob.glob(os.path.join(dir_path, "*t2w.png")))
            else:
                files = sorted(glob.glob(os.path.join(dir_path, "*.png")))
        
            # Store the image and label paths in the dictionary
            data_paths[f'{dataset}_{data_type.split("_")[1]}'] = files

    print('Number of training images', len(data_paths['train_images']))
    print('Number of validation images', len(data_paths['val_images']))
    print('Number of test images', len(data_paths['test_images']))
    return data_paths


def training(args): 
    ckp_path = args.ckp_path
    if not os.path.exists(ckp_path):
        os.makedirs(ckp_path)

    processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
    model = SamWithTextPrompt(sam_type=args.sam_type)
    batch_size = args.batch_size
    best_val_loss = 100
    #model.load_state_dict(torch.load('./checkpoints/best_weights.pth'))
    # create train and validation dataloaders
    datasets = ['train', 'val', 'test']
    data_types = ['2d_images_cat', '2d_masks']

    data_paths = creat_datasets(datasets, args.data_root, data_types)
    print(len(data_paths['train_images']), len(data_paths['train_masks']))
    train_dataset = SAMDataset(image_paths=data_paths['train_images'], mask_paths=data_paths['train_masks'], 
                               processor=processor, text_prompt=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = SAMDataset(image_paths=data_paths['val_images'], mask_paths=data_paths['val_masks'],
                            processor=processor, text_prompt=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # define training loop
    num_epochs = args.num_epoch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # print(model)
    trainable_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params}")
    # define optimizer
    # optimizer = Adam(model.sam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # define segmentation loss with sigmoid activation applied to predictions from the model
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # track mean train and validation losses
    mean_train_losses, mean_val_losses = [], []


    # set model to train mode for gradient updating
    model.train()
    with torch.no_grad():
    
        # create temporary list to record training losses
        epoch_losses = []
        for i in range(len(train_dataset)):

            # forward pass
            batch_source = train_dataset.__getitem__(i)
            batch_target = val_dataset.__getitem__(i)
            text_prompt = generate_prompts()
            print(text_prompt)
            # print( batch["pixel_values"].shape, batch["ground_truth_mask"].shape,)
            src_pred_msk, src_boxes, src_phrases, src_logits = model.predict(image_pil=batch_source["pixel_values"],
                      text_prompt=text_prompt, input_boxes = batch_source["input_boxes"])
            tgt_pred_msk, tgt_boxes, tgt_phrases, tgt_logits = model.predict(image_pil=batch_target["pixel_values"],
                      text_prompt=text_prompt, input_boxes = batch_target["input_boxes"])
            # compute loss
            labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(src_phrases, src_logits)]
            ground_truth_masks = batch_source["ground_truth_mask"].float()
            src_img = np.asarray(batch_source["pixel_values"])
            src_img = torch.from_numpy(src_img).permute(2,0,1)
            
            # print(src_pred_msk.shape, src_boxes.shape, src_img.shape) # shape [1, 256, 256] [1, 4] [3, 256, 256]
            src_img = draw_image(src_img, src_pred_msk, src_boxes, labels) #torch.Tensor(batch_source['input_boxes'][0]), ['prompt'])#
            
            labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(tgt_phrases, tgt_logits)]
            print(labels)
            tgt_img = np.asarray(batch_target["pixel_values"])
            tgt_img = torch.from_numpy(tgt_img).permute(2,0,1)
            tgt_img = draw_image(tgt_img, tgt_pred_msk, tgt_boxes, labels)
            if src_pred_msk.shape[0]>0 and tgt_pred_msk.shape[0]>0:
                fig, axs = plt.subplots(2, 2, figsize=(10, 5))
                max_idx_src = torch.argmax(src_logits)
                max_idx_tgt = torch.argmax(tgt_logits)
                axs[0,0].imshow(src_pred_msk[max_idx_src,...])
                axs[0,0].axis('off')
                axs[0,1].imshow(src_img.permute(1,2,0))
                axs[0,1].axis('off')
                axs[1,0].imshow(tgt_pred_msk[max_idx_tgt,...])
                axs[1,0].axis('off')
                axs[1,1].imshow(tgt_img.permute(1,2,0))
                axs[1,1].axis('off')
                plt.tight_layout()
                plt.savefig('debug1.png')
                plt.close()

 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--ckp_path', type=str, default='checkpoints')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--model_name', type=str, default='samregnet')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--data_root', type=str, default='datasets')
    parser.add_argument('--continue_train', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


    training(args)
