import os
import glob
import monai
import torch
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt
from dataloaders.LangDataLoader3d import dataset_loaders
from torch.utils.data import DataLoader
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
import cv2


def dice_score(prediction, target):
    smooth = 1e-10 # Smoothing factor to prevent division by zero
    intersection = torch.sum(prediction * target, dim=tuple(range(1, len(prediction.shape))))
    union = torch.sum(prediction, dim=tuple(range(1, len(prediction.shape)))) + torch.sum(target, dim=tuple(range(1, len(prediction.shape))))
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.mean() # Return the dice score as a scalar value

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

   
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

def set_input(source, idx, ch=3):
    source = source[:,:,idx].numpy()
    if ch>1:
        source = np.stack([source] * ch ,axis=-1)
    source = (source*255).astype(np.uint8)
    src_input = Image.fromarray(source).resize((1000,1000))
    return src_input

def prepare_draw_image(batch_source, src_pred_msk, src_boxes, labels):
    src_img = np.asarray(batch_source)
    src_img = torch.from_numpy(src_img).permute(2,0,1)
    src_img = draw_image(src_img, src_pred_msk, src_boxes, labels) #torch.Tensor(batch_source['input_boxes'][0]), ['prompt'])#
    return src_img   

def check_overlap(bbox, mask):
    a, b, c, d = torch.round(bbox).int()
    h, w = mask.shape
    mask = mask.astype(bool)
    # Create a binary mask for the bounding box
    bbox_mask = np.zeros((h, w), dtype=bool)
    bbox_mask[b:d, a:c] = True  # Assign True to the bounding box area
    # Check for overlap using logical AND
    overlap = np.logical_and(bbox_mask, mask)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(bbox_mask)
    ax[1].imshow(mask)
    ax[2].imshow(overlap)
    plt.savefig('bbox_mask.png')
    plt.close()
    if not np.any(overlap):
        return False
    else:
        return True if np.sum(overlap)/np.sum(mask)>0.5 else False
        
    
    # return np.any(overlap)  # Returns True if there is any overlap

def calculate_size(rois):
    rois_size = []
    for roi in rois:
        roi[roi>0]=1
        rois_size.append(torch.sum(roi))
    rois_size = np.stack(rois_size, axis=0)
    return np.min(rois_size), np.max(rois_size), np.mean(rois_size)
        
    
def training(args):
    # Set seed for deterministic behavior
    set_seed(42)
    fid = open(f'{args.ckp_path}/testlog.txt', 'w')
    ckp_path = args.ckp_path
    savefigs = os.path.join(args.ckp_path, args.savefigs)
    os.makedirs(savefigs, exist_ok=True)
    if not os.path.exists(ckp_path):
        os.makedirs(ckp_path)

    # processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
    model = SamWithTextPrompt(sam_type=args.sam_type)
    batch_size = args.batch_size

    train_dataset = dataset_loaders(path=args.dataroot, phase='train', batch_size=batch_size, np_var='vol', add_feat_axis=True)
    val_dataset = dataset_loaders(path=args.dataroot, phase='valid', batch_size=batch_size, np_var='vol',  add_feat_axis=True)
   
   
    # define training loop
    num_epochs = args.num_epoch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # print(model)
    trainable_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params}")
    # define optimizer


    # set model to train mode for gradient updating
    model.train()
    epoch = 0
    corres=0
    total=0
    src=0
    tgt=0
    # with torch.no_grad():
    if True:
        # create temporary list to record training losses
        epoch_losses = []
        epoch+=1
        for i in range(num_epochs):
            # forward pass
            case_acc = []
            # forward pass
            input_dict= val_dataset.__getitem__(i)
            batch_target, batch_source = input_dict['fx_img'], input_dict['mv_img'] # [batch, 1, x, y, z]
            tgt_seg, src_seg = input_dict['fx_seg'], input_dict['mv_seg']
            
           
            text_prompt = 'big'#, black in left, black in right, bladder in upper middle, rectum, bone, tumor, hole, vessel, fat, high signal, low signal, muscle'
            min_len = min(batch_target.shape[0],batch_source.shape[0])
            # idx = np.random.randint(min_len//2-10, min_len//2+10)
            print(text_prompt)
            all_rois_small, all_rois_large, all_rois_mean=0,0,0
            for idx in range(min_len):#(min_len-20, min_len+20, 2):
                src_input = Image.fromarray(batch_source[idx])
                tgt_input = Image.fromarray(batch_target[idx])
                
                if not np.any(src_seg[idx]):
                     continue
                total+=1
                roi_matching = RoiMatching(src_input, tgt_input)
                with torch.no_grad():
                    src_pred_msk, src_boxes, src_phrases, src_logits, src_emb = model.predict(image_pil=src_input,
                      text_prompt=text_prompt)
  
                    tgt_pred_msk, tgt_boxes, tgt_phrases, tgt_logits, tgt_emb  = model.predict(image_pil=tgt_input,
                      text_prompt=text_prompt)
                    
                    src_paired_roi, tgt_paired_roi = roi_matching.get_paired_roi(src_pred_msk.numpy(), tgt_pred_msk.numpy(), src_emb, tgt_emb)
                    
                    if len(src_paired_roi)==0:
                        continue

                    rois_small, rois_large, rois_mean = calculate_size(src_paired_roi)
                    total += 1
                    corres += len(src_paired_roi)
                    src += len(src_pred_msk)
                    tgt += len(tgt_pred_msk)
                    
                    all_rois_small+=rois_small
                    all_rois_large+=rois_large
                    all_rois_mean+=rois_mean
                    
                    


                
            print(f"src: {src}, tgt: {tgt}, corres: {corres}")     
        print(f"small: {all_rois_small/total}, large: {all_rois_large/total}, mean: {all_rois_mean/total}")             
            # print(f"case{i} done")
    # print(f"Correspondence for {text_prompt}: {corres}/{total}")
                    
                    
                            
                        
                
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--ckp_path', type=str, default='checkpoints')
    parser.add_argument('--savefigs', type=str, default='savefigs_multi')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--model_name', type=str, default='samregnet')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dataroot', type=str, default='datasets')
    parser.add_argument('--continue_train', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


    training(args)
