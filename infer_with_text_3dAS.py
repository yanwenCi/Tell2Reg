import os
import glob
import monai
import torch
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt
from dataloaders import AS_dataloader as dataloaders
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
    class_num = prediction.shape[0]
    prediction_flat = prediction.reshape(class_num, -1)
    target_flat = target.reshape(class_num, -1)

    intersection = torch.sum(prediction_flat * target_flat, dim=-1)
    union = torch.sum(prediction_flat, dim=-1) + torch.sum(target_flat, dim=-1)
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.mean().item()  # Return the dice score as a scalar value

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

# creat dataset
def set_dataloader(data_root):
        train_set = dataloaders.LongitudinalData(data_root, phase='train')
        
        val_set = dataloaders.LongitudinalData(data_root, phase='val')
        
        print('>>> train set ready. length:', len(train_set))
        test_set = dataloaders.LongitudinalData(data_root, phase='test')
        
        return train_set, val_set, test_set
   
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

def plot_together(src_img, tgt_img, src_paired_roi, tgt_paired_roi, masks_warped, image_warped, i, idx):
                fig, axs = plt.subplots(3,2, figsize=(5, 5))
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
                plt.savefig(f'savefigs/debug{i}_{idx}.png')
                plt.close()

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
    
    
    train_dataset = dataloaders.LongitudinalData(args.data_root, phase='train')
    
   
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
    epoch = 0
    # with torch.no_grad():
    while epoch<num_epochs:
        # create temporary list to record training losses
        epoch_losses = []
        epoch+=1
        for i, input_dict in enumerate(train_dataset):
            # forward pass
            case_acc = []
            
            batch_target, batch_source = input_dict['fx_img'], input_dict['mv_img'] # [batch, 1, x, y, z]
            tgt_seg, src_seg = input_dict['fx_seg'], input_dict['mv_seg']
            
            case_stack = torch.zeros_like(tgt_seg)

            text_prompt = ['hole','edge','head', 'prostate', 'texture','middle','bathrobe', 'cloth']#, black in left, black in right, bladder in upper middle, rectum, bone, tumor, hole, vessel, fat, high signal, low signal, muscle'
            min_len = min(batch_target.shape[-1],batch_source.shape[-1])
            # idx = np.random.randint(min_len//2-10, min_len//2+10)

            for idx in range(min_len):#(min_len-20, min_len+20, 2):
                src_input = set_input(batch_source, idx)
                tgt_input = set_input(batch_target, idx)
                src_seg_in = set_input(src_seg, idx, ch=1)
                tgt_seg_in = set_input(tgt_seg, idx, ch=1)
                with torch.no_grad():
                    src_pred_msk, src_boxes, src_phrases, src_logits, src_emb = model.predict(image_pil=src_input,
                      text_prompt=text_prompt)
                    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(src_phrases, src_logits)]
                    print('source', labels)
  
                    tgt_pred_msk, tgt_boxes, tgt_phrases, tgt_logits, tgt_emb = model.predict(image_pil=tgt_input,
                      text_prompt=text_prompt)
                    tgt_labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(tgt_phrases, tgt_logits)]
                    print('target', tgt_labels)
                    print('before paring', src_pred_msk.shape, tgt_pred_msk.shape)
                
                src_img = prepare_draw_image(src_input, src_pred_msk, src_boxes, labels)
                tgt_img = prepare_draw_image(tgt_input, tgt_pred_msk, tgt_boxes, tgt_labels)

                cv2.imwrite(f'savefigs/src{i}_{idx}.png', src_img.permute(1,2,0).numpy())
                cv2.imwrite(f'savefigs/tgt{i}_{idx}.png', tgt_img.permute(1,2,0).numpy())
                roi_matching = RoiMatching(src_input, tgt_input)
                if src_pred_msk.shape[0]>0 and tgt_pred_msk.shape[0]>0:
                    # save warped ROIs for visulisation
           
                    src_paired_roi, tgt_paired_roi = roi_matching.get_paired_roi(src_pred_msk.numpy(), tgt_pred_msk.numpy(), src_emb, tgt_emb)
                    print('after pairing', src_paired_roi.shape, tgt_paired_roi.shape)
            
                    paired_rois = PairedRegions(masks_mov=src_paired_roi, masks_fix=tgt_paired_roi,  device=device)
                    ddf = paired_rois.get_dense_correspondence(transform_type='ddf', max_iter=int(1e4), lr=1e-3, w_ddf=1000.0, verbose=True)
                    
                    # warp the source image and mask to the target image space
                    masks_warped = (warp_by_ddf(src_paired_roi.to(dtype=torch.float32, device=device), ddf)).to(torch.uint8).cpu()
                    image_warped = (warp_by_ddf(to_tensor(src_input).to(dtype=torch.float32, device=device), ddf)*255).to(torch.uint8).cpu()
                    # real task of prostate segmentation
                    prostate_mask_warped = (warp_by_ddf(to_tensor(src_seg_in).to(dtype=torch.float32, device=device), ddf)).to(torch.uint8).cpu()
                    
                    # print(masks_warped.max(), tgt_paired_roi.max(), prostate_mask_warped.max(), to_tensor(tgt_seg_in).max()) 255 1 255 1
                    prostate_mask_warped=F.interpolate(prostate_mask_warped[None,...], size=(128, 128), mode='bilinear', align_corners=False)
                   
                    case_stack[:,:,idx]=prostate_mask_warped[0,0]
                    dice=dice_score(masks_warped, tgt_paired_roi)
                    print(f'Dice score: {dice}')
                    epoch_losses.append(dice)
                   
                    #plot and save images
                    if idx == min_len//2:
                        for k in range(src_paired_roi.shape[0]):
                            cv2.imwrite(f'savefigs/src_paired_roi{i}_{idx}_{k}.png', src_paired_roi[k,...].numpy().astype(np.uint8)*255)
                        for k in range(tgt_paired_roi.shape[0]):
                            cv2.imwrite(f'savefigs/tgt_paired_roi{i}_{idx}_{k}.png', tgt_paired_roi[k,...].numpy().astype(np.uint8)*255)
                        for k in range(masks_warped.shape[0]):
                            cv2.imwrite(f'savefigs/masks_warped{i}_{idx}_{k}.png', masks_warped[k,...].numpy()*255)
                        cv2.imwrite(f'savefigs/image_warped{i}_{idx}.png', image_warped.permute(1,2,0).numpy())
                    
                        plot_together(src_img, tgt_img, src_paired_roi, tgt_paired_roi, masks_warped, image_warped, i, idx)
            
            case_dice = dice_score(case_stack[None,...], tgt_seg[None,...])
            case_acc.append(case_dice)
            print(f'Case {i} Dice score: {case_dice}')

    print(f'Overall Dice score: {np.mean(np.stack(epoch_losses, axis=0), axis=0)}, Case Dice score: {np.mean(np.stack(case_acc, axis=0), axis=0)}')

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

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    training(args)
