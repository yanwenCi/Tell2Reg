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
from dataloaders.LangDataLoader3d import dataset_loaders
from networks.networks import SamWithTextPrompt, draw_image
# from networks.networks_with_pretrain import SamWithTextPrompt
from text_prompts import  generate_prompts
from PIL import Image
#matplotlib inline
from region_correspondence.region_correspondence.paired_regions import PairedRegions
from region_correspondence.region_correspondence.utils import warp_by_ddf
from networks.paired_roi import RoiMatching
import torchvision.transforms as transforms
import random

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

def sorted_indices(tgt, ref):
    sorted_indices = sorted(range(len(ref)), key=lambda k: ref[k])[::-1]

    return tgt[sorted_indices, ...], ref[sorted_indices]

def to_tensor(image):
    to_tensor_func = transforms.ToTensor()
    return to_tensor_func(image)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def training(args):
    # Set seed for deterministic behavior
    set_seed(42)
    ckp_path = args.ckp_path
    if not os.path.exists(ckp_path):
        os.makedirs(ckp_path)

    # processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
    model = SamWithTextPrompt(sam_type=args.sam_type)
    sam = model.build_sam()
    batch_size = args.batch_size
    best_val_loss = 100
    #model.load_state_dict(torch.load('./checkpoints/best_weights.pth'))
    # create train and validation dataloaders
    datasets = ['train', 'val', 'test']
    data_types = ['2d_images_cat', '2d_masks']
    
    
    train_dataset = dataset_loaders(path=args.data_root, phase='train', batch_size=batch_size, np_var='vol', add_feat_axis=True)
    val_dataset = dataset_loaders(path=args.data_root, phase='valid', batch_size=batch_size, np_var='vol',  add_feat_axis=True)
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
    # with torch.no_grad():
    while True:
        # create temporary list to record training losses
        epoch_losses = []
        for i in range(len(train_dataset)):

            # forward pass
            batch_source= train_dataset.__getitem__(i)
            batch_target= val_dataset.__getitem__(i)
            # text_prompt = generate_prompts()
            # print(text_prompt)
            text_prompt = 'ball'#, black in left, black in right, bladder in upper middle, rectum, bone, tumor, hole, vessel, fat, high signal, low signal, muscle'
            min_len = min(batch_target.shape[0],batch_source.shape[0])//2
            idx = np.random.randint(min_len-10, min_len+10)
            # print( batch["pixel_values"].shape, batch["ground_truth_mask"].shape,)
            src_input = Image.fromarray(batch_source[idx])
            tgt_input = Image.fromarray(batch_target[idx])
            with torch.no_grad():
                src_pred_msk, src_boxes, src_phrases, src_logits = model.predict(image_pil=src_input,
                      text_prompt=text_prompt)
                labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(src_phrases, src_logits)]
                print(labels)
                
                # src_pred_msk, src_boxes, src_phrases, src_logits = model.predict(image_pil=src_input,
                #       text_prompt=text_prompt)
                # labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(src_phrases, src_logits)]
                # print(labels)

                tgt_pred_msk, tgt_boxes, tgt_phrases, tgt_logits = model.predict(image_pil=tgt_input,
                      text_prompt=text_prompt)
                tgt_labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(tgt_phrases, tgt_logits)]
                print(tgt_labels)
            # use everything model
            # print('src_pred_msk', src_pred_msk.shape, src_boxes.shape)
            # tgt_pred_msk = model.pred_everything_sam(tgt_input)
            # src_pred_msk = model.pred_everything_sam(src_input)
            # tgt_pred_msk = torch.from_numpy(np.stack(tgt_pred_msk['masks'], axis=0))
            # src_pred_msk = torch.from_numpy(np.stack(src_pred_msk['masks'], axis=0))
            # need simplified
            
            # ground_truth_masks = batch_source["ground_truth_mask"].float()
            src_img = np.asarray(batch_source[idx])
            src_img = torch.from_numpy(src_img).permute(2,0,1)
            # print(src_pred_msk.shape, src_boxes.shape, src_img.shape) # shape [1, 256, 256] [1, 4] [3, 256, 256]
            src_img = draw_image(src_img, src_pred_msk, src_boxes, labels) #torch.Tensor(batch_source['input_boxes'][0]), ['prompt'])#
            

            tgt_img = np.asarray(batch_target[idx])
            tgt_img = torch.from_numpy(tgt_img).permute(2,0,1)
            tgt_img = draw_image(tgt_img, tgt_pred_msk, tgt_boxes, tgt_labels)


            # estimate dense correspondence
            print('before paring', src_pred_msk.shape, tgt_pred_msk.shape)
            # src_pred_msk, src_logits = sorted_indices(src_pred_msk, src_logits)
            # tgt_pred_msk, tgt_logits = sorted_indices(tgt_pred_msk, tgt_logits)

            # len_msk = min(src_pred_msk.shape[0], tgt_pred_msk.shape[0])
            # src_pred_msk = src_pred_msk[:len_msk, ...]
            # tgt_pred_msk = tgt_pred_msk[:len_msk, ...]
            # src_logits = src_logits[:len_msk]
            # tgt_logits = tgt_logits[:len_msk]
            

            roi_matching = RoiMatching(src_input, tgt_input)
            if src_pred_msk.shape[0]>0 and tgt_pred_msk.shape[0]>0:
                # save warped ROIs for visulisation
                # src_paired_roi, tgt_paired_roi = _overlap_pair(src_pred_msk, tgt_pred_msk)
                src_paired_roi, tgt_paired_roi = roi_matching.get_paired_roi(src_pred_msk.numpy(), tgt_pred_msk.numpy())
                print('after pairing', src_paired_roi.shape, tgt_paired_roi.shape)
                paired_rois = PairedRegions(masks_mov=src_paired_roi, masks_fix=tgt_paired_roi, device=device)
                ddf = paired_rois.get_dense_correspondence(transform_type='ddf', max_iter=int(1e4), lr=1e-3, w_ddf=100.0, verbose=True)
          
                masks_warped = (warp_by_ddf(src_paired_roi.to(dtype=torch.float32, device=device), ddf)*255).to(torch.uint8)
                image_warped = (warp_by_ddf(to_tensor(src_input).to(dtype=torch.float32, device=device), ddf)*255).to(torch.uint8)
              
                fig, axs = plt.subplots(3,2, figsize=(5, 5))
                # max_idx_src = torch.argmax(src_logits)
                # max_idx_tgt = torch.argmax(tgt_logits)
                # axs[0,0].imshow(src_pred_msk[0,...])
                # axs[0,0].axis('off')
                axs[0,0].imshow(src_img.permute(1,2,0), cmap='gray')
                axs[0,0].axis('off')
                axs[0,0].set_title('Source')
                # axs[0,1].imshow(tgt_pred_msk[0,...])
                # axs[1,0].axis('off')
                axs[0,1].imshow(tgt_img.permute(1,2,0))
                axs[0,1].axis('off')
                axs[0,1].set_title('Target')
                axs[1,0].imshow(src_paired_roi[0,...].cpu().detach().numpy(), cmap='gray')
                axs[1,0].axis('off')
                axs[1,0].set_title('1st Source Paired ROI')
                axs[1,1].imshow(tgt_paired_roi[0,...].cpu().detach().numpy(), cmap='gray')
                axs[1,1].axis('off')
                axs[1,1].set_title('1st Target Paired ROI')
                axs[2,0].imshow(masks_warped[0,...].cpu().detach().numpy(), cmap='gray')
                axs[2,0].axis('off')
                axs[2,0].set_title('Warped 1st Source ROI')
                axs[2,1].imshow(image_warped[0,...].cpu().detach().numpy(), cmap='gray')
                axs[2,1].axis('off')
                axs[2,1].set_title('Warped -1st Source ROI')
                plt.tight_layout()
                plt.savefig(f'savefigs/debug{i}.png')
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
