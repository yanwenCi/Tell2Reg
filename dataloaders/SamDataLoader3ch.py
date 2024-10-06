import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    Compose,
    CropForegroundd,
    CopyItemsd,
    LoadImaged,
    CenterSpatialCropd,
    Invertd,
    OneOf,
    Orientationd,
    MapTransform,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    CenterSpatialCropd,
    RandSpatialCropd,
    SpatialPadd,
    ScaleIntensityRanged,
    Spacingd,
    RepeatChanneld,
    ToTensord,
)

def get_bounding_box(ground_truth_map):
    '''
    This function creates varying bounding box coordinates based on the segmentation contours as prompt for the SAM model
    The padding is random int values between 5 and 20 pixels
    '''

    if len(np.unique(ground_truth_map)) > 1:

        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(5, 20))
        x_max = min(W, x_max + np.random.randint(5, 20))
        y_min = max(0, y_min - np.random.randint(5, 20))
        y_max = min(H, y_max + np.random.randint(5, 20))
        
        bbox = [x_min, y_min, x_max, y_max]

        return bbox
    else:
        return [0, 0, 256, 256] # if there is no mask in the array, set bbox to image size
        
 
class SAMDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor, roi_size=(256, 256), text_prompt=None):
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.text_prompt = text_prompt
        self.transforms = Compose([
            
            # load .nii or .nii.gz files
            LoadImaged(keys=['img', 'img1', 'img2', 'label']),
            
            # add channel id to match PyTorch configurations
            EnsureChannelFirstd(keys=['img', 'img1', 'img2', 'label']),
            
            # reorient images for consistency and visualization
            Orientationd(keys=['img', 'img1', 'img2', 'label'], axcodes='RA'),
            
            # resample all training images to a fixed spacing
            Spacingd(keys=['img', 'img1', 'img2', 'label'], pixdim=(0.5, 0.5), mode=("bilinear","bilinear", "bilinear", "nearest")),
            
            # rescale image and label dimensions to 256x256 
            CenterSpatialCropd(keys=['img', 'img1', 'img2', 'label'], roi_size=roi_size),
            
            # scale intensities to 0 and 255 to match the expected input intensity range
            ScaleIntensityRanged(keys=['img', 'img1', 'img2'], a_min=-1000, a_max=2000, 
                         b_min=0.0, b_max=255.0, clip=True), 
            
            ScaleIntensityRanged(keys=['label'], a_min=0, a_max=255, 
                         b_min=0.0, b_max=1.0, clip=True), 

            SpatialPadd(keys=["img", "img1", "img2", "label"], spatial_size=roi_size)
#             RepeatChanneld(keys=['img'], repeats=3, allow_missing_keys=True)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # create a dict of images and labels to apply Monai's dictionary transforms
        data_dict = self.transforms({'img': image_path, 'img1': image_path.replace('t2w', 'adc'), 'img2': 
                                     image_path.replace('t2w', 'dwi'), 'label': mask_path})

        # squeeze extra dimensions
        image = data_dict['img'].squeeze()
        image1 = data_dict['img1'].squeeze()
        image2 = data_dict['img2'].squeeze()
        ground_truth_mask = data_dict['label'].squeeze()

        # convert to int type for huggingface's models expected inputs
        image = image.astype(np.uint8)
        image1 = image1.astype(np.uint8)
        image2 = image2.astype(np.uint8)
        #print(image.max(), image.min(), image.mean())
        # convert the grayscale array to RGB (3 channels)
       
        array_rgb = np.dstack((image, image, image))
        # print(array_rgb.shape, array_rgb.max(), ground_truth_mask.shape)
        # convert to PIL image to match the expected input of processor
        image_rgb = Image.fromarray(array_rgb)
         
        # get bounding box prompt (returns xmin, ymin, xmax, ymax)
        # in this dataset, the contours are -1 so we change them to 1 for label and 0 for background
        # ground_truth_mask[ground_truth_mask < 0] = 1
        
        prompt = get_bounding_box(ground_truth_mask)
        # print(prompt)
        # prepare image and prompt for the model
        inputs = self.processor(image_rgb, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        # print(inputs['pixel_values'].min(), inputs['pixel_values'].max(), inputs['pixel_values'].mean())
        # add ground truth segmentation (ground truth image size is 256x256)
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask.astype(np.int8))

        text_prompt = 'lesion'#self.describe_bounding_box(256, 256, *prompt)
        if self.text_prompt is not None:
            inputs["text_prompts"] = text_prompt
        return inputs
    
    def describe_bounding_box(self, image_width, image_height, x, y, width, height):
        # Calculate center of the bounding box
        center_x = x + width / 2
        center_y = y + height / 2

        # Describe position relative to image edges
        if center_x < image_width / 3:
            horizontal_position = "left"
        elif center_x < 2 * image_width / 3:
            horizontal_position = "center"
        else:
            horizontal_position = "right"

        if center_y < image_height / 3:
            vertical_position = "top"
        elif center_y < 2 * image_height / 3:
            vertical_position = "middle"
        else:
            vertical_position = "bottom"

        # Describe position relative to image center
        if center_x < image_width / 2:
            horizontal_relative_position = "left"
        else:
            horizontal_relative_position = "right"

        if center_y < image_height / 2:
            vertical_relative_position = "above"
        else:
            vertical_relative_position = "below"

        # Construct description
        description = f"The bounding box is located in the {horizontal_position} {vertical_position} of the image.It is {horizontal_relative_position} {abs(center_x - image_width / 2):.2f} pixels from the image center horizontally, and {vertical_relative_position} {abs(center_y - image_height / 2):.2f} pixels from the image center vertically."

        return description



