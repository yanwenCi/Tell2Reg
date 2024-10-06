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
from SamDataLoader3ch import SAMDataset


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
    
def build_sam(model_name='facebook/sam-vit-base'):
    # create an instance of the processor for image preprocessing
    processor = SamProcessor.from_pretrained(model_name)
    print(processor)

    # load the pretrained weights for finetuning
    model = SamModel.from_pretrained(model_name)

    # make sure we only compute gradients for mask decoder (encoder weights are frozen)
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False) 
    return model, processor

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
    model, processor = build_sam(model_name=args.model_name)
    batch_size = args.batch_size
    best_val_loss = 100
    if args.continue_train:
        model.load_state_dict(torch.load('./checkpoints/best_weights.pth'))
    # create train and validation dataloaders
    datasets = ['train', 'val', 'test']
    data_types = ['2d_images_cat', '2d_masks']

    data_paths = creat_datasets(datasets, args.data_root, data_types)
    print(len(data_paths['train_images']), len(data_paths['train_masks']))
    train_dataset = SAMDataset(image_paths=data_paths['train_images'], mask_paths=data_paths['train_masks'], processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = SAMDataset(image_paths=data_paths['val_images'], mask_paths=data_paths['val_masks'], processor=processor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # define training loop
    num_epochs = args.num_epoch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(model)
    trainable_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params}")
    # define optimizer
    optimizer = Adam(model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # define segmentation loss with sigmoid activation applied to predictions from the model
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # track mean train and validation losses
    mean_train_losses, mean_val_losses = [], []


    # set model to train mode for gradient updating
    model.train()
    for epoch in range(num_epochs):
    
        # create temporary list to record training losses
        epoch_losses = []
        for i, batch in enumerate(tqdm(train_dataloader)):

            # forward pass
            # print( batch["pixel_values"].shape, batch["ground_truth_mask"].shape,)
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)
            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            # print(batch["pixel_values"].shape, predicted_masks.shape)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            # print(dice_score(predicted_masks, ground_truth_masks.unsqueeze(1)))
            #print(ground_truth_masks.max(),batch['pixel_values'].min(), batch['pixel_values'].max(), batch['pixel_values'].mean())
            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())
        
            # visualize training predictions every 50 iterations
            if i % 1000 == 0:
            
                # clear jupyter cell output
                clear_output(wait=True)
            
                fig, axs = plt.subplots(1, 3)
                xmin, ymin, xmax, ymax = get_bounding_box(batch['ground_truth_mask'][0])
                rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')

                axs[0].set_title('input image')
                axs[0].imshow(batch["pixel_values"][0,0], cmap='gray')
                axs[0].axis('off')

                axs[1].set_title('ground truth mask')
                axs[1].imshow(batch['ground_truth_mask'][0], cmap='copper')
                axs[1].add_patch(rect)
                axs[1].axis('off')
            
                # apply sigmoid
                medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))

                        # convert soft mask to hard mask
                medsam_seg_prob = medsam_seg_prob.detach().cpu().numpy().squeeze()
                medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

                axs[2].set_title('predicted mask')
                axs[2].imshow(medsam_seg[0,...], cmap='copper')
                axs[2].axis('off')

                plt.tight_layout()
                #plt.show()
                plt.savefig(f'./training_figures/epoch_{epoch}_iter_{i}.png')
                plt.close()



        val_losses, val_dices = validation(model, device, val_dataloader, seg_loss, epoch)
        # save the best weights and record the best performing epoch
        if mean(val_losses) < best_val_loss:
            torch.save(model.state_dict(), f"./{ckp_path}/best_weights.pth")
            print(f"Model Was Saved! Current Best val loss {best_val_loss:.4f}, epoch {epoch}")
            best_val_loss = mean(val_losses)


        else:
            print(mean(val_losses), best_val_loss)
            print("Model Was Not Saved!")


        print(f'EPOCH: {epoch}, Mean loss: {mean(epoch_losses):.4f}, mean dice score: {mean(val_dices):.4f}')
        mean_train_losses.append(mean(epoch_losses))
        mean_val_losses.append(mean(val_losses))




def validation(model, device, val_dataloader, seg_loss, epoch):
    # create temporary list to record validation losses
    val_losses = []
    val_dices = []
    # set model to eval mode for validation
    with torch.no_grad():
        for j, val_batch in enumerate(tqdm(val_dataloader)):
            
            # forward pass
            outputs = model(pixel_values=val_batch["pixel_values"].to(device),
                      input_boxes=val_batch["input_boxes"].to(device),
                      multimask_output=False)
            
            # calculate val loss
            predicted_val_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = val_batch["ground_truth_mask"].float().to(device)
            val_loss = seg_loss(predicted_val_masks, ground_truth_masks.unsqueeze(1))
            val_losses.append(val_loss.item())
            val_dices.append(dice_score(predicted_val_masks, ground_truth_masks.unsqueeze(1)))
            
        
            # visualize the last validation prediction
            if j%500 == 0:
                fig, axs = plt.subplots(1, 3)
                xmin, ymin, xmax, ymax = get_bounding_box(val_batch['ground_truth_mask'][0])
                rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
                axs[0].set_title('input image')
                axs[0].imshow(val_batch["pixel_values"][0,1], cmap='gray')
                axs[0].axis('off')

                axs[1].set_title('ground truth mask')
                axs[1].imshow(val_batch['ground_truth_mask'][0], cmap='copper')
                axs[1].add_patch(rect)
                axs[1].axis('off')

                # apply sigmoid
                medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
                
                # convert soft mask to hard mask
                medsam_seg_prob = medsam_seg_prob.detach().cpu().numpy().squeeze()
                medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
                

                axs[2].set_title('predicted mask')
                axs[2].imshow(medsam_seg[0], cmap='copper')
                axs[2].axis('off')

                plt.tight_layout()
                #plt.show()
                plt.savefig(f'./test_figures/epoch_{epoch}_iter_{j}.png')
                plt.close()

        
    return val_losses, val_dices
 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--ckp_path', type=str, default='checkpoints/samseg')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--data_root', type=str, default='datasets')
    parser.add_argument('--model_name', type=str, default='facebook/sam-vit-base')
    parser.add_argument('--continue_train', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


    training(args)
