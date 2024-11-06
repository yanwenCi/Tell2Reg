import os.path as osp
import SimpleITK as sitk
import os
import numpy as np
import cv2 

data_dir = 'datasets/data_lesion_cross1'
data_list = open(osp.join(data_dir, 'train', 'pair_path_list.txt')).readlines()
images0 = [data_list[i].strip().split(' ')[0] for i in range(len(data_list))]
labels = [data_list[i].strip().split(' ')[2] for i in range(len(data_list))]
images1 = [data_list[i].strip().split(' ')[3] for i in range(len(data_list))]
images2 = [data_list[i].strip().split(' ')[4] for i in range(len(data_list))]


print('No. of images:', len(images0), ' labels:', len(labels))

base_dir = './datasets'
datasets = ['train', 'val', 'test']
#data_types = ['2d_proimages', '2d_promasks']
data_types = ['2d_images_cat', '2d_masks']

# Create directories
dir_paths = {}
for dataset in datasets:
    for data_type in data_types:
        # Construct the directory path
        dir_path = os.path.join(base_dir, f'{dataset}_{data_type}')
        dir_paths[f'{dataset}_{data_type}'] = dir_path
        # Create the directory
        os.makedirs(dir_path, exist_ok=True)


def normalize_sitk(image_path):
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    min_value = image_array.min()
    max_value = image_array.max()

    # Define the new range (0 to 256)
    new_min = 0
    new_max = 256

    # Normalize the pixel values to the new range
    normalized_image = sitk.Cast((image - min_value) * (new_max - new_min) / (max_value - min_value) + new_min, sitk.sitkUInt8)
    print(sitk.GetArrayFromImage(normalized_image).max())
    return normalized_image


# Assuming first 2 patients for training, next 1 for validation and last 1 for testing
p = 0
for idx, (img_path, mask_path) in enumerate(zip(images0, labels)):
    # Load the 3D image and mask
    # img0 = sitk.ReadImage(img_path)
    # if 'cat' in data_types[0]:
    #     img1 = sitk.ReadImage(images1[idx])
    #     img2 = sitk.ReadImage(images2[idx])
    # mask = sitk.ReadImage(mask_path)*256

    img0 = normalize_sitk(img_path)
    if 'cat' in data_types[0]:
        img1 = normalize_sitk(images1[idx])
        img2 = normalize_sitk(images2[idx])
    mask = sitk.ReadImage(mask_path)*256
    #print(sitk.GetArrayFromImage(img0).max(), sitk.GetArrayFromImage(img0).min() , sitk.GetArrayFromImage(mask).max())
    print('processing patient', idx, img0.GetSize(), mask.GetSize())

    # Get the mask data as numpy array
    mask_data = sitk.GetArrayFromImage(mask)
    #print(mask_data.shape)
    # Select appropriate directories
    if idx % 5 < 3 :  # Training
        img_dir = dir_paths[f'train_{data_types[0]}']
        mask_dir = dir_paths[f'train_{data_types[1]}']
    elif idx % 5  == 3:  # Validation
        img_dir = dir_paths[f'val_{data_types[0]}']
        mask_dir = dir_paths[f'val_{data_types[1]}']
    else:  # Testing
        img_dir = dir_paths[f'test_{data_types[0]}']
        mask_dir = dir_paths[f'test_{data_types[1]}']

    # Iterate over the axial slices

    for i in range(img0.GetSize()[-1]):
        # If the mask slice is not empty, save the image and mask slices

        if np.any(mask_data[i, :, :]):
            # Prepare the new ITK images
            # img_slice = img[i, :, :]
            # mask_slice = mask[i, :, :]
            img_slice = img0[:,:,i]
            mask_slice = mask[:,:,i]
            
            # Define the output paths
            img_slice_path = os.path.join(img_dir, f"{os.path.basename(img_path).replace('.nii.gz', '')}_{i}.png")
            mask_slice_path = os.path.join(mask_dir, f"{os.path.basename(mask_path).replace('.nii.gz', '')}_{i}.png")

            # Save the slices as NIfTI files
            
            # sitk.WriteImage(mask_slice, mask_slice_path)
            cv2.imwrite(mask_slice_path, sitk.GetArrayViewFromImage(mask_slice))
            if 'cat' in data_types[0]:
                # sitk.WriteImage(img_slice, img_slice_path.replace('.nii.gz', '_t2w.nii.gz'))
                # sitk.WriteImage(img1[:,:,i], img_slice_path.replace('.nii.gz', '_adc.nii.gz'))
                # sitk.WriteImage(img2[:,:,i], img_slice_path.replace('.nii.gz', '_dwi.nii.gz'))
                cv2.imwrite(img_slice_path.replace('.png', '_t2w.png'), sitk.GetArrayViewFromImage(img_slice))
                cv2.imwrite(img_slice_path.replace('.png', '_adc.png'), sitk.GetArrayViewFromImage(img1[:,:,i]))
                cv2.imwrite(img_slice_path.replace('.png', '_dwi.png'), sitk.GetArrayViewFromImage(img2[:,:,i]))
            else:
                sitk.WriteImage(img_slice, img_slice_path)#.replace('.nii.gz', '_t2w.nii.gz'))
    p += i       
print('total slices', p , 'train:test:val=', len(images0)*0.6, len(images0)*0.2, len(images0)*0.2)
# Initialize dictionary for storing image and label paths
data_paths = {}
