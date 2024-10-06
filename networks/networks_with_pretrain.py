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
# for pretrained models
from transformers import SamModel 
from transformers import SamProcessor
import numpy as np

# def draw_image(image, masks, boxes, labels, alpha=0.4):
#     if len(boxes) > 0:
#         image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
#     if len(masks) > 0:
#         image = draw_segmentation_masks(image, masks=masks, colors=['cyan'] * len(masks), alpha=alpha)

#     return image

SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
 
CACHE_PATH = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth"))
CACHE_PATH = "/raid/candi/Wen/SamMedImg/checkpoints/samseg/best_weights.pth"

class SamWithTextPrompt(nn.Module):
    def __init__(self, sam_type='vit_h', return_prompts=False, ckpt_path=CACHE_PATH, device='gpu'):
        super(SamWithTextPrompt, self).__init__()
        self.sam_type = sam_type
        self.build_groundingdino()
        self.return_prompts = return_prompts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_sam(ckpt_path)
       
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

    # def build_sam(self, ckpt_path):
    #     if self.sam_type is None or ckpt_path is None:
    #         if self.sam_type is None:
    #             print("No sam type indicated. Using vit_h by default.")
    #             self.sam_type = "vit_h"
    #         checkpoint_url = SAM_MODELS[self.sam_type]
    #         try:
    #             sam = sam_model_registry[self.sam_type]()
    #             state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
    #             sam.load_state_dict(state_dict, strict=True)
    #         except:
    #             raise ValueError(f"Problem loading SAM please make sure you have the right model type: {self.sam_type} \
    #                 and a working checkpoint: {checkpoint_url}. Recommend deleting the checkpoint and \
    #                 re-downloading it.")
    #         sam.to(device=self.device)
    #         self.sam = SamPredictor(sam)
    #     else:
    #         try:
    #             sam = sam_model_registry[self.sam_type](ckpt_path)
    #         except:
    #             raise ValueError(f"Problem loading SAM. Your model type: {self.sam_type} \
    #             should match your checkpoint path: {ckpt_path}. Recommend calling LangSAM \
    #             using matching model type AND checkpoint path")
    #         sam.to(device=self.device)
    #         self.sam = SamPredictor(sam)
    def build_sam(self, ckpt_path, model_name='facebook/sam-vit-base'):
        self.processor = SamProcessor.from_pretrained(model_name)
        self.sam = SamModel.from_pretrained(model_name)
        self.sam.load_state_dict(torch.load(ckpt_path))

    def build_groundingdino(self):
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = self.load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)

    def predict_dino(self, image_pil, text_prompt, box_threshold, text_threshold):
        image_array = self.transform_image(image_pil)
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_array,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         remove_combined=self.return_prompts,
                                         device=self.device)
        
        W, H = image_array.size()[-2:]
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
  
        return boxes, logits, phrases

    def predict_sam(self, image_pil, boxes):
        # boxes = [box[None,None, :] for box in boxes]
        # boxes = torch.cat(boxes, dim=1)
        # print(boxes.shape)
        inputs = self.processor(image_pil, input_boxes=[[[boxes]]], return_tensors="pt")
        # print(inputs["pixel_values"].shape, inputs["input_boxes"].shape)
        # self.sam.set_image(image_array)
        # transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        outputs = self.sam(pixel_values=inputs["pixel_values"].to(self.device),
                      input_boxes=inputs["input_boxes"].to(self.device),multimask_output=False)
        predicted_masks = outputs.pred_masks.squeeze(1)
        medsam_seg_prob = torch.sigmoid(predicted_masks)
        medsam_seg_prob = medsam_seg_prob[0]
        medsam_seg = (medsam_seg_prob > 0.5)
        # print(predicted_masks.shape)
        return medsam_seg.cpu()

    def predict(self, image_pil, text_prompt, input_boxes=None, box_threshold=0.3, text_threshold=0.5):
        boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
        # masks = torch.tensor([])
        # if len(boxes) > 0:
        #     masks = self.predict_sam(image_pil, boxes)
        #     masks = masks.squeeze(1)
        masks = self.predict_sam(image_pil, input_boxes)
        return masks, boxes, phrases, logits