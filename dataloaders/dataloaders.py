import torch.utils.data as data
import pickle as pkl
import os
import torch
import random
import numpy as np
from glob import glob
from scipy.ndimage import zoom

torch.set_default_tensor_type('torch.FloatTensor')


class LongitudinalData(data.Dataset):
    def __init__(self, data_path, key_file, phase, patient_cohort = 'inter+intra'):
        self.phase = phase
        assert phase in ['train', 'val', 'test'], "phase cannot recongnise..."
        self.patient_cohort = patient_cohort
        self.data_path = data_path
        self.key_file = os.path.join(self.data_path, key_file)
        self.key_pairs_list = self.get_key_pairs()
        self.image_folder = 'images'

    def __getitem__(self, index):
        ## the code of sampling strategies can be further optimized
        if index == 0:
            self.key_pairs_list = self.get_key_pairs()
      
        moving_key, fixed_key = self.key_pairs_list[index]
        moving_image, moving_label = np.load(os.path.join(self.data_path, self.image_folder, moving_key + '-T2.npy'))
        fixed_image, fixed_label = np.load(os.path.join(self.data_path, self.image_folder, fixed_key + '-T2.npy'))
        moving_image, fixed_image = self._cat_rgb(moving_image), self._cat_rgb(fixed_image)
        fixed_label, moving_label = self._normalize255(fixed_label), self._normalize255(moving_label)
        data_dict = {
            'mv_img': moving_image.astype(np.uint8), 
            'mv_seg':moving_label.astype(np.uint8), 
            'fx_img': fixed_image.astype(np.uint8), 
            'fx_seg': fixed_label.astype(np.uint8),
            'mv_key': moving_key,
            'fx_key': fixed_key,
            }

        if self.phase != 'test':
            return data_dict
        else:
            mv_ldmk_paths = glob(os.path.join(self.data_path, self.image_folder, moving_key + '-T2-ldmark*'))
            mv_ldmk_paths.sort(key=lambda x: int(os.path.basename(x).replace('.npy', '').split('-')[-1]))
            mv_ldmk_arrs = [torch.FloatTensor(np.load(i)) for i in mv_ldmk_paths]

            fx_ldmk_paths = glob(os.path.join(self.data_path, self.image_folder, fixed_key + '-T2-ldmark*'))
            fx_ldmk_paths.sort(key=lambda x: int(os.path.basename(x).replace('.npy', '').split('-')[-1]))
            fx_ldmk_arrs = [torch.FloatTensor(np.load(i)) for i in fx_ldmk_paths]
            # print(mv_ldmk_paths, fx_ldmk_paths)
            data_dict['mv_ldmk_paths'] = mv_ldmk_paths
            data_dict['mv_ldmks'] = mv_ldmk_arrs
            data_dict['fx_ldmk_paths'] = fx_ldmk_paths
            data_dict['fx_ldmks'] = fx_ldmk_arrs
            
            return data_dict 

    def _normalize255(self, arr): 
        arr = zoom(arr, (1000/128, 1000/128, 1), order=1)
        arr = arr.transpose(2,1,0)
        return (arr - arr.min())/(arr.max() - arr.min()) * 255
           
    def _cat_rgb(self, tmp):
        tmp = self._normalize255(tmp)
        tmp=np.concatenate([tmp[...,None]]*3, axis=-1)
        # tmp=np.rot90(tmp, k=1, axes=(1,2))
        return tmp
    

    def __len__(self):
        return len(self.key_pairs_list)

    def get_key_pairs(self):
        '''
        have to manually define shuffling rules.
        '''
        with open(self.key_file, 'rb') as f:
            key_dict = pkl.load(f)
        l = key_dict[self.phase]
        if self.phase == 'train':
            if self.patient_cohort == 'intra':
                l = self.__odd_even_shuffle__(l)
            elif self.patient_cohort == 'inter':
                l = self.__get_inter_patient_pairs__(l)
            elif self.patient_cohort == 'inter+intra':
                l1 = self.__odd_even_shuffle__(l)
                l2 = self.__get_inter_patient_pairs__(l)
                l3 = self.__inter_lock__(l1, l2)
                l = l3[:len(l)]
            elif self.patient_cohort == 'ex+inter+intra':
                l1 = self.__odd_even_shuffle__(l)
                l2 = self.__get_inter_patient_pairs__(l, extra=key_dict['extra'])
                l3 = self.__inter_lock__(l1, l2)
                l = l3[:len(l)]
            else:
                print('wrong patient cohort.')
        return l

    def __get_inter_patient_pairs__(self, l, extra = None):
        k = [i[0] for i in l]  # get all images
        k = list(set(k))  # get rid of repeat keys
        if extra is not None:
            assert type(extra) == list, "extra should be a list contains key values."
            k += extra
        else: pass 
        l = [(i, j) for i in k for j in k]  # get all combinations
        l = [i for i in l if i[0].split('-')[0] != i[1].split('-')[0]]  # exclude same patient
        random.shuffle(l)
        tmp = l[:len(k)]
        return tmp  # get the same length as random ordered dataloader

    @staticmethod
    def __inter_lock__(l1, l2):
        new_list = []
        for a, b in zip(l1, l2):
            new_list.append(a)
            new_list.append(b)
        return new_list

    def __odd_even_shuffle__(self, l):
        even_list, odd_list, new_list = [], [], []
        for idx, i in enumerate(l):
            if (idx % 2) == 0:
                even_list.append(i)
            else:
                odd_list.append(i)
        random.shuffle(even_list)
        random.shuffle(odd_list)
        new_list = self.__inter_lock__(even_list, odd_list)
        return new_list

