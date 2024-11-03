import torch
from torch import nn
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
import os 
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
import groundingdino.datasets.transforms as T
import numpy as np
import random
from transformers import pipeline, SamModel, SamProcessor

def draw_image(image, masks, boxes, labels, alpha=0.4):
    # print(boxes)
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(boxes))]
    if len(boxes) > 0:
        image = draw_bounding_boxes(image, boxes, colors=colors, labels=labels, width=5)
    if len(masks) > 0:
        image = draw_segmentation_masks(image, masks=masks, colors=colors, alpha=alpha)

    return image

SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
 
CACHE_PATH = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth"))


class SamWithTextPrompt(nn.Module):
    def __init__(self, sam_type='vit_h', return_prompts=False, ckpt_path=CACHE_PATH, device='gpu'):
        super(SamWithTextPrompt, self).__init__()
        self.sam_type = sam_type
        self.build_groundingdino()
        self.return_prompts = return_prompts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_sam()
     
       
       
    def load_model_hf(self, repo_id, filename, ckpt_config_filename):
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

        args = SLConfig.fromfile(cache_config_file)
        model = build_model(args)

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location='cpu')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print(f"Model loaded from {cache_file} \n => {log}")
        model.eval()
        return model
    
    def transform_image(self, image) -> torch.Tensor:
        transform = T.Compose([
        # T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(image, None)
        return image_transformed

    def build_sam(self, ckpt_path=None):
        if self.sam_type is None or ckpt_path is None:
            if self.sam_type is None:
                print("No sam type indicated. Using vit_h by default.")
                self.sam_type = "vit_h"
            checkpoint_url = SAM_MODELS[self.sam_type]
            try:
                sam = sam_model_registry[self.sam_type]()
                state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
                sam.load_state_dict(state_dict, strict=True)
            except:
                raise ValueError(f"Problem loading SAM please make sure you have the right model type: {self.sam_type} \
                    and a working checkpoint: {checkpoint_url}. Recommend deleting the checkpoint and \
                    re-downloading it.")
            sam.to(device=self.device)
            self.sam = SamPredictor(sam)
        else:
            try:
                sam = sam_model_registry[self.sam_type](ckpt_path)
            except:
                raise ValueError(f"Problem loading SAM. Your model type: {self.sam_type} \
                should match your checkpoint path: {ckpt_path}. Recommend calling LangSAM \
                using matching model type AND checkpoint path")
            sam.to(device=self.device)
            self.sam = SamPredictor(sam)
        return self.sam

    def pred_everything_sam(self, imgs):
        generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=self.device)
        if self.sam_type=='sam_h':
                outputs = generator(imgs, points_per_batch=64,pred_iou_thresh=0.90,stability_score_thresh=0.9,)
                # outputs = generator(imgs, points_per_batch=64,pred_iou_thresh=0.70,stability_score_thresh=0.7,)

        elif self.sam_type == 'medsam':
                # outputs = generator(imgs, points_per_batch=64,stability_score_thresh=0.7,) #medsam
                outputs = generator(imgs, points_per_batch=64,stability_score_thresh=0.8,) #medsam

        else:
                outputs = generator(imgs, points_per_batch=64, stability_score_thresh=0.9, )
        return outputs
    
    def build_groundingdino(self):
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = self.load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)

    def predict_dino(self, image_pil, text_prompt, box_threshold, text_threshold):
        image_array = self.transform_image(image_pil)
        # image_array = torch.FloatTensor(image_array).to(self.device)
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_array,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         #remove_combined=self.return_prompts,
                                         device=self.device)
        
        W, H = image_array.size()[-2:]
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
  
        return boxes, logits, phrases

    def predict_sam(self, image_pil, boxes=None):
        image_array = np.asarray(image_pil)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2]).to(self.sam.device) if boxes is not None else None
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        image_embeddings = self.sam.get_image_embedding() 
        return masks.cpu(), image_embeddings
    
    def filter_full_image_bboxes(self, bboxes, logits, phrases, image_width, image_height):
        full_image_bboxes, filtered_logits, filtered_phrases = [], [], []
        # print(bboxes)
        for bbox, logit, phrase in zip(bboxes, logits, phrases):
            x_min, y_min, x_max, y_max = bbox

            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            if bbox_width < image_width-20 and bbox_height < image_height-20:
                full_image_bboxes.append(bbox)
                filtered_logits.append(logit)
                filtered_phrases.append(phrase)
        # print(full_image_bboxes)
        if len(full_image_bboxes) == 0:
            return torch.tensor([]), torch.tensor([]), []
        else:
            return torch.stack(full_image_bboxes), torch.stack(filtered_logits), filtered_phrases
    
    def predict(self, image_pil, text_prompt, box_threshold=0.15, text_threshold=0.15, input_boxes=None):
        if isinstance(text_prompt, str):
            boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
        elif isinstance(text_prompt, list):
            boxes, logits, phrases = [], [], []
            for prompt in text_prompt:
                box, logit, phrase = self.predict_dino(image_pil, prompt, box_threshold, text_threshold)
                boxes.extend(box)
                logits.extend(logit)
                phrases.extend(phrase)
            boxes=torch.stack(boxes, dim=0)
            logits=torch.stack(logits, dim=0)
                

        masks = torch.tensor([])
        embeddings = torch.tensor([])
        boxes, logits, phrases = self.filter_full_image_bboxes(boxes, logits, phrases, image_pil.size[0], image_pil.size[1])
        
        if len(boxes) > 0:
            masks, embeddings = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)
            # print('masks shape befor:', masks.shape)
            masks, boxes, phrases, logits = self._mask_criteria(masks, boxes, phrases, logits)
            # print('masks shape after:', masks.shape)
        return masks, boxes, phrases, logits, embeddings
    
    def _mask_criteria(self, masks, boxes, phrases, logits, v_min=2000, v_max= 5e5, overlap_ratio=0.8):
        remove_list = set()
        for _i, mask in enumerate(masks):
            if mask.sum() < v_min or mask.sum() > v_max:
                remove_list.add(_i)
                
        masks = [mask for idx, mask in enumerate(masks) if idx not in remove_list]
        boxes = [box for idx, box in enumerate(boxes) if idx not in remove_list]
        phrases = [phrase for idx, phrase in enumerate(phrases) if idx not in remove_list]
        logits = [logit for idx, logit in enumerate(logits) if idx not in remove_list]
        
        n = len(masks)
        remove_list = set()
        for i in range(n):
            for j in range(i + 1, n):
                mask1, mask2 = masks[i], masks[j]
                intersection = (mask1 & mask2).sum()
                smaller_mask_area = min(masks[i].sum(), masks[j].sum())

                if smaller_mask_area > 0 and (intersection / smaller_mask_area) >= overlap_ratio:
                    if mask1.sum() < mask2.sum():
                        remove_list.add(i)
                    else:
                        remove_list.add(j)
                    # print('remove mask {} or mask {}'.format(i, j))
        masks = [mask for idx, mask in enumerate(masks) if idx not in remove_list]
        boxes = [box for idx, box in enumerate(boxes) if idx not in remove_list]
        phrases = [phrase for idx, phrase in enumerate(phrases) if idx not in remove_list]
        logits = [logit for idx, logit in enumerate(logits) if idx not in remove_list]
        
        if len(masks) == 0:
            return torch.tensor([]), torch.tensor([]), [], []
        else:
            masks = torch.stack(masks)
            boxes = torch.stack(boxes)  
            return masks, boxes, phrases, logits
    
