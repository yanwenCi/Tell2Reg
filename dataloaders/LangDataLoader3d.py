import os
import numpy as np
import dataloaders.utils as utils
import torch.utils.data as data
import torchio as tio


class dataset_loaders(data.Dataset):
    def __init__(self, path, 
                 phase, batch_size=1, 
                 np_var='vol', add_batch_axis=False, pad_shape=None,
                resize_factor=(2.5,2.5,1), add_feat_axis=False, crop_size=None,
                istest=False, transform=None):
        self.path = path
        self.phase = phase
        self.batch_size = batch_size
        self.np_var = np_var
        self.pad_shape = pad_shape
        self.resize_factor = resize_factor
        self.add_feat_axis = add_feat_axis
        self.add_batch_axis = add_batch_axis
        self.crop_size = crop_size
        self.istest = istest
        self.transforms = transform
        self.path_list=open(os.path.join(self.path, self.phase, 'pair_path_list.txt'), 'r').readlines()
        self.t2w_filenames = [x.strip().split(' ')[0] for x in self.path_list]
        self.adc_filenames = [x.strip().split(' ')[3] for x in self.path_list]
        self.dwi_filenames = [x.strip().split(' ')[4] for x in self.path_list]
        self.msk_filenames = [x.strip().split(' ')[1] for x in self.path_list]
        self.zon_filenames = [x.strip().split(' ')[2] for x in self.path_list]
        print(f"data length: {len(self.t2w_filenames)}")

    def norm255(self, image):
        img=255*(image - image.min())/(image.max() - image.min())
        return img.astype(np.uint8)
    
    def load_data(self, indices):
    # convert glob path to filenames

        
        if len(self.t2w_filenames) != len(self.msk_filenames):
            raise ValueError('Number of image files must match number of seg files.')

        #for i in range(len(self.t2w_filenames)):
        
            #indices = [j for j in range(i, min((i+self.batch_size),len(self.t2w_filenames)))]
        vols=[]
        names=[]
        # load volumes and concatenate
        load_params = dict(np_var=self.np_var, add_batch_axis=self.add_batch_axis, add_feat_axis=self.add_feat_axis,
                           pad_shape=self.pad_shape, resize_factor=self.resize_factor, crop_size=self.crop_size)
        # for vol_names in [self.t2w_filenames, self.dwi_filenames, self.adc_filenames]:
        #     vols.append(self.norm255(utils.load_volfile(vol_names[indices], **load_params)))
        # names.append(self.t2w_filenames[indices].split('/')[-1])
        vols = utils.load_volfile(self.t2w_filenames[indices], **load_params)
        vols = self.norm255(vols)
        return vols

        # if self.istest:
        #     load_params['np_var'] = 'seg'  # be sure to load seg
        #     vols.append(utils.load_volfile(self.msk_filenames[indices], **load_params) )

        #     load_params['np_var'] = 'seg'  # be sure to load seg
        #     vols.append(utils.load_volfile(self.zon_filenames[indices], **load_params) )

        #     return tuple(vols+names)

        # else:
        #     load_params['np_var'] = 'seg'  # be sure to load seg
        #     vols.append(utils.load_volfile(self.msk_filenames[indices], **load_params))
        #     return tuple(vols)
        
    def __len__(self):
        return len(self.t2w_filenames)
    

    def slicing(self, img):
        img = img.transpose(2, 1, 0, 3) #z, y, x, c
        img = np.concatenate((img, img, img), axis=-1)
        img = [img[i] for i in range(0, img.shape[0]) if img[i].sum() > 0]
        img = np.stack(img)

        return img

    def __getitem__(self, idx):
        if self.istest:
            scan1 = self.load_data(idx)
            # outvols = msk.transpose(2, 1,0, 3)
            invols = self.slicing(scan1)
            return invols#, outvols, zone, name)
        else:
            scan1 = self.load_data(idx)
            # outvols = msk.transpose(2, 1,0, 3)
            invols = self.slicing(scan1)
            return invols#, outvols)
        

   
        