import torch.utils.data as data
import pickle as pkl
import os
import torch
import random
import numpy as np
from glob import glob
from scipy.ndimage import zoom

torch.set_default_tensor_type('torch.FloatTensor')


class DiffusionData(data.Dataset):
    def __init__(self, config, phase):
        self.phase = phase
        assert phase in ['train', 'val', 'test'], "phase cannot recongnise..."

        self.config = config
        self.data_path = self.config.data_path
        self.key_file = os.path.join(self.data_path, self.config.key_file)
        print(f'data_file exists: {os.path.exists(self.key_file)}')
        self.key_pairs_list = self.get_key_pairs()
        self.image_folder = 'images'
        # self.pre_load()

    def pre_load(self):
        self.images=[]
        for index in range(len(self.key_pairs_list)):
            moving_key, fixed_key = self.key_pairs_list[index]
            fixed_image, fixed_label = np.load(os.path.join(self.data_path, self.image_folder, fixed_key + '-T2.npy'))
            moving_image, moving_label = np.load(os.path.join(self.data_path, self.image_folder, moving_key + '-T2.npy'))
            self.images.append(moving_image.transpose(2, 1, 0))
            self.images.append(fixed_image.transpose(2,1, 0))
        self.images = np.concatenate(self.images, axis=0)
        #print(f'images shape: {self.images.shape}')
            


    def __getitem__(self, index):
        # the code of sampling strategies can be further optimized
        # print(f'index: {index}')
        if index == 0:
            self.key_pairs_list = self.get_key_pairs()

        moving_key, fixed_key = self.key_pairs_list[index]
        fixed_image, fixed_label = np.load(os.path.join(self.data_path, self.image_folder, fixed_key + '-T2.npy'))
        moving_image, moving_label = np.load(os.path.join(self.data_path, self.image_folder, moving_key + '-T2.npy'))

        return fixed_image
        # slices = self.images[index]
        # return slices

            
               

    def __len__(self):
        return len(self.key_pairs_list)
        # leng = self.images.shape[0]
        # return leng

    def get_key_pairs(self):
        '''
        have to manually define shuffling rules.
        '''
        with open(self.key_file, 'rb') as f:
            key_dict = pkl.load(f)
        l = key_dict[self.phase]
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

    

class LongitudinalData(data.Dataset):
    def __init__(self, data_path, phase):
        self.phase = phase
        assert phase in ['train', 'val', 'test'], "phase cannot recongnise..."

        self.data_path = data_path
        self.key_file = os.path.join(self.data_path, 'key-train-IFIB-val-IFIB-test-IFIB.pkl' )
        self.patient_cohort = 'intra+inter'
        self.key_pairs_list = self.get_key_pairs()
        self.image_folder = 'images'
        

    def __getitem__(self, index):
        ## the code of sampling strategies can be further optimized
        if index == 0:
            self.key_pairs_list = self.get_key_pairs()
        moving_key, fixed_key = self.key_pairs_list[index]
        moving_image, moving_label = np.load(os.path.join(self.data_path, self.image_folder, moving_key + '-T2.npy'))
        fixed_image, fixed_label = np.load(os.path.join(self.data_path, self.image_folder, fixed_key + '-T2.npy'))
        
        # if self.config.patched and (self.phase == 'train'):
        #     moving_image, moving_label, fixed_image, fixed_label = pre.random_crop_3d([moving_image, moving_label, fixed_image, fixed_label], self.config.patch_size)
        
        # if self.config.cropped:
        #     moving_image, moving_label, fixed_image, fixed_label = pre.random_crop_3d([moving_image, moving_label, fixed_image, fixed_label], self.config.crop_size)
       
        data_dict = {
            'mv_img': torch.FloatTensor(moving_image).rot90(k=1, dims=(0,1)), 
            'mv_seg': torch.FloatTensor(moving_label).rot90(k=1, dims=(0,1)), 
            'fx_img': torch.FloatTensor(fixed_image).rot90(k=1, dims=(0,1)), 
            'fx_seg': torch.FloatTensor(fixed_label).rot90(k=1, dims=(0,1)),
            'mv_key': moving_key,
            'fx_key': fixed_key,
            }

        if self.phase != 'test':
            return data_dict
        else:
            mv_ldmk_paths = glob(os.path.join(self.data_path, self.image_folder, moving_key + '-T2-ldmark*'))
            mv_ldmk_paths.sort(key=lambda x: int(os.path.basename(x).replace('.npy', '').split('-')[-1]))
            mv_ldmk_arrs = [torch.FloatTensor(np.load(i)).rot90(k=1, dims=(0,1)) for i in mv_ldmk_paths]

            fx_ldmk_paths = glob(os.path.join(self.data_path, self.image_folder, fixed_key + '-T2-ldmark*'))
            fx_ldmk_paths.sort(key=lambda x: int(os.path.basename(x).replace('.npy', '').split('-')[-1]))
            fx_ldmk_arrs = [torch.FloatTensor(np.load(i)).rot90(k=1, dims=(0,1)) for i in fx_ldmk_paths]
            # print(mv_ldmk_paths, fx_ldmk_paths)
            data_dict['mv_ldmk_paths'] = mv_ldmk_paths
            data_dict['mv_ldmks'] = mv_ldmk_arrs
            data_dict['fx_ldmk_paths'] = fx_ldmk_paths
            data_dict['fx_ldmks'] = fx_ldmk_arrs
            
            return data_dict        

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


class mpMRIData(data.Dataset):
    def __init__(self, config, phase, external=None):
        assert phase in ['train', 'val', 'test'], "phase incorrect..."
        self.phase = phase
        self.config = config
        self.is_external = False

        if (external is not None) and phase=='test':
            self.is_external = True
            self.data_path = external
            self.data_list = glob(os.path.join(self.data_path, '*'))
        else:
            self.data_path = self.data_path
            self.data_list = glob(os.path.join(self.data_path, self.phase, '*'))

        self.data_list.sort()
        self.mi_record_list = self.data_list.copy()


    def __getitem__(self, index):
        data = {}
        image_path = self.data_list[index]
        mod = ['t2', 'dwi_b0', 'dwi']
        for m in mod:
            if self.is_external and m=='dwi_b0':
                matrix = np.load(os.path.join(image_path, 'dwi.npy'))         
                data[m] = torch.FloatTensor(self.normalize(matrix[None, ...]))
                data[f'{m}_path'] = os.path.join(image_path, 'dwi.npy')
                continue
            matrix = np.load(os.path.join(image_path, f'{m}.npy'))
            data[m] = torch.FloatTensor(self.normalize(matrix[None, ...]))
            data[f'{m}_path'] = os.path.join(image_path, f'{m}.npy')

        if self.phase=='train' and self.method=='mixed':
            '''0.5 probability to use dwi_b0 and 0.5 for dwi as moving image'''
            if self.rand_prob(0.5):
                # print('mixed mode')
                data['dwi'] = torch.clone(data['dwi_b0'])

        if self.phase == 'test':
            data.update(self.__get_landmarks__(image_path))
        return data

    def __get_landmarks__(self, image_path):
        t2_ldmk_paths = glob(os.path.join(image_path, 't2_ldmk_*'))
        dwi_ldmk_paths = [i.replace('t2_ldmk_', 'dwi_ldmk_') for i in t2_ldmk_paths]
        
        t2_ldmk_paths.sort()
        dwi_ldmk_paths.sort()

        t2_ldmks = [self.normalize(np.load(i)) for i in t2_ldmk_paths]
        t2_ldmks = torch.FloatTensor(np.stack(t2_ldmks))
        dwi_ldmks = [self.normalize(np.load(i)) for i in dwi_ldmk_paths]
        dwi_ldmks = torch.FloatTensor(np.stack(dwi_ldmks))
        return {'t2_ldmks': t2_ldmks, 'dwi_ldmks':dwi_ldmks, 't2_ldmks_paths':t2_ldmk_paths, 'dwi_ldmks_paths':dwi_ldmk_paths}

    def __dump_mi_record__(self, idx, tensor_arr, ori_patient_path):
        pid = os.path.basename(ori_patient_path)
        dataset_name = os.path.basename(self.data_path)
        tmp_dataset_name = f"{dataset_name}-{self.exp_name}"
        save_folder = os.path.join(os.path.dirname(self.data_path), tmp_dataset_name, pid)
        os.makedirs(save_folder, exist_ok=True)
        self.mi_record_list[idx] = save_folder  # update new path for transformed dwi_b0
        
        save_name = os.path.join(save_folder, 'dwi_b0.npy')
        print(save_name)
        np.save(save_name, torch.squeeze(tensor_arr).numpy())

    @staticmethod
    def normalize(arr):
        '''normalize to 0-1'''
        return (arr - arr.min())/(arr.max() - arr.min())

    @staticmethod
    def rand_prob(p=0.5):
        assert 0<=p<=1, "p should be a number in [0, 1]"
        return random.random() < p
        
    def __len__(self):
        return len(self.data_list)


class CBCTData(data.Dataset):
    def __init__(self, config, phase):
        self.phase = phase
        assert phase in ['train', 'val', 'test'], "phase cannot recongnise..."
        self.config = config
        self.data_path = self.config.data_path
        self.data_pairs = self.__get_data_pairs__()
        self.PC = config.patient_cohort
        assert self.PC in ['inter', 'intra'], f"patient_cohort should be intra/inter, cannot be {self.PC}"

    def __getitem__(self, index):
        flag = (self.phase=='train') and (self.rand_prob()) and (self.config.patient_cohort=='inter')
        if flag:
            fx_img_path, _, fx_seg_path, _ = self.data_pairs[index]
            _, mv_img_path, _, mv_seg_path = self.__get_inter_pairs__(index)
        else:
            fx_img_path, mv_img_path, fx_seg_path, mv_seg_path = self.data_pairs[index]

        moving_image, moving_label = np.load(mv_img_path), np.load(mv_seg_path)
        fixed_image, fixed_label = np.load(fx_img_path), np.load(fx_seg_path)

        if self.phase in ['train'] and self.config.two_stage_sampling:
            random_label_index = random.randint(0, moving_label.shape[0]-1)
            moving_label, fixed_label = moving_label[random_label_index], fixed_label[random_label_index]
        else: pass

        if self.phase in ['train'] and self.config.crop_on_seg_aug and self.rand_prob():
            moving_label = self.random_crop_aug(moving_label)
            
        data_dict = {
            'mv_img': torch.FloatTensor(moving_image[None, ...]), 
            'mv_seg': torch.FloatTensor(moving_label[None, ...]), 
            'fx_img': torch.FloatTensor(fixed_image[None, ...]), 
            'fx_seg': torch.FloatTensor(fixed_label[None, ...]),
            'subject': os.path.basename(fx_img_path),
            'subject_mv': os.path.basename(mv_img_path),
            }

        return data_dict

    def __len__(self):
        return len(self.data_pairs)

    def __get_data_pairs__(self):
        '''split train val test data'''
        pid_lists = os.listdir(os.path.join(self.data_path, 'fixed_images'))
        pid_lists.sort()

        tmp = []
        for i in pid_lists:
            tmp.append([
                os.path.join(self.data_path, 'fixed_images', i),
                os.path.join(self.data_path, 'moving_images', i),
                os.path.join(self.data_path, 'fixed_labels', i),
                os.path.join(self.data_path, 'moving_labels', i),
            ])
        
        test_pairs = tmp[len(pid_lists)//8 * self.config.cv : len(pid_lists)//8 * (self.config.cv+1)]

        if self.config.cv == 7:
            val_pairs = tmp[:len(pid_lists)//8]
        else:
            val_pairs = tmp[len(pid_lists)//8 * (self.config.cv+1) : len(pid_lists)//8 * (self.config.cv+2)]
        
        train_pairs = [i for i in tmp if i not in val_pairs and i not in test_pairs]

        data_dict = {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
        
        return data_dict[self.phase]

    def __get_inter_pairs__(self, index):
        indices = list(range(self.__len__()))
        del indices[index]
        return self.data_pairs[random.sample(indices, 1)[0]]

    @staticmethod
    def rand_prob(p=0.5):
        assert 0<=p<=1, "p should be a number in [0, 1]"
        return random.random() < p

    def random_crop_aug(self, seg_arr):
        '''A data augmentation method for conditional segmentation'''
        px, py, pz = np.where(seg_arr==1)
        grid_points = [i for i in zip(px, py, pz)]
        cx, cy, cz = random.sample(grid_points, 1)[0]  # select a point as center

        r_min, r_max = self.config.crop_on_seg_rad
        shape_x, shape_y, shape_z = self.config.input_shape

        rad = random.randint(r_min, r_max)
        Lx, Rx = max(0, cx-rad), min(shape_x, cx+rad)  # Left & Right x
        Ly, Ry = max(0, cy-rad), min(shape_y, cy+rad)  # Left & Right y
        Lz, Rz = max(0, cz-rad), min(shape_z, cz+rad)  # Left & Right z

        seg_arr[..., Lx:Rx, Ly:Ry, Lz:Rz] = 0
        return seg_arr
        
        
        
