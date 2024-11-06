import os
import numpy as np
import torch.utils.data as data
import torchio as tio
import PIL.Image as Image
import scipy

class dataset_loaders(data.Dataset):
    def __init__(self, path, 
                 phase, batch_size=1, 
                 np_var='vol', add_batch_axis=False, pad_shape=None,
                resize_factor=(1000, 1000, 0.5), add_feat_axis=False, crop_size=None,
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
        files = os.listdir(path)
        self.pair_filenames = [os.path.join(path, f) for f in files if 'mask' not in f]
        print(self.pair_filenames)                   
        self.zone_filenames = [f.replace('.nii.gz', '_mask.nii.gz') for f in self.pair_filenames]
        print(f"data length: {len(self.pair_filenames)}")

    def norm255(self, image):
        img=255*(image - image.min())/(image.max() - image.min())
        return img.astype(np.uint8)
    
    def load_data(self, indices):
        load_params = dict(np_var=self.np_var, add_batch_axis=self.add_batch_axis, add_feat_axis=self.add_feat_axis,
                           pad_shape=self.pad_shape, resize_factor=self.resize_factor, crop_size=self.crop_size)
        vols =[]
        for l in range(2): 
            tmp=load_volfile(self.pair_filenames[l], **load_params).transpose(2,0,1,3)
            tmp = self.norm255(tmp)
            tmp=np.concatenate([tmp]*3, axis=-1)
            tmp=np.rot90(tmp, k=1, axes=(1,2))
            vols.append(tmp)
               
        segs = [load_volfile(self.zone_filenames[0], **load_params),
                load_volfile(self.zone_filenames[1], **load_params)]
        
        segs = [self.binarize(seg) for seg in segs]
        return vols, segs

    def binarize(self, seg):
        seg[seg>0] = 255
        seg = seg.transpose(2,0,1, 3)
        seg = np.rot90(seg, k=1, axes=(1,2))
        return seg.astype(np.uint8)
    
    def __len__(self):
        return len(self.pair_filenames)//2
 

    def __getitem__(self, idx):
        dict_data = {}
        scan, seg = self.load_data(idx)
        dict_data = {'mv_img': scan[0], 'mv_seg': seg[0], 'fx_img': scan[1], 'fx_seg': seg[0],
                     'fx_key': self.pair_filenames[0].split('/')[-1], 'mv_key': self.pair_filenames[1].split('/')[-1]}
        return dict_data
 
        

   
def load_volfile(
    filename,
    np_var='vol',
    add_batch_axis=False,
    add_feat_axis=False,
    pad_shape=None,
    resize_factor=1,
    ret_affine=False,
    crop_size=None,
):
    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format. If input file is not a string,
    returns it directly (allows files preloaded in memory to be passed to a generator)

    Parameters:
        filename: Filename to load, or preloaded volume to be returned.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    if isinstance(filename, str) and not os.path.isfile(filename):
        raise ValueError("'%s' is not a file." % filename)

    if not os.path.isfile(filename):
        if ret_affine:
            (vol, affine) = filename
        else:
            vol = filename
    elif filename.endswith(('.nii', '.nii.gz', '.mgz')):
        import nibabel as nib
        img = nib.load(filename)
        vol = img.get_fdata().squeeze()
        affine = img.affine
    elif filename.endswith('.npy'):
        vol = np.load(filename)
        affine = None
    elif filename.endswith('.npz'):
        npz = np.load(filename)
        vol = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
        affine = None
    else:
        raise ValueError('unknown filetype for %s' % filename)

    if add_feat_axis:
        vol = vol[..., np.newaxis]

    # if resize_factor != 1:
    #     vol = resize(vol, resize_factor)
    if resize_factor[0]/vol.shape[0] != 1:
        # print(f"resize factor: {resize_factor}")
        resize_factor = (resize_factor[0]/vol.shape[0], resize_factor[1]/vol.shape[1], resize_factor[2])
        vol = resize(vol, resize_factor)
        # print(f"vol shape: {vol.shape}")

    if pad_shape:
        vol, _ = pad(vol, pad_shape)

    if crop_size is not None:
        vol = crop(vol, crop_size)

    if add_batch_axis:
        vol = vol[np.newaxis, ...]

    return (vol, affine) if ret_affine else vol.astype(np.float32)


def save_volfile(array, filename, affine=None):
    """
    Saves an array to nii, nii.gz, or npz format.

    Parameters:
        array: The array to save.
        filename: Filename to save to.
        affine: Affine vox-to-ras matrix. Saves LIA matrix if None (default).
    """
    if filename.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        if affine is None and array.ndim >= 3:
            # use LIA transform as default affine
            affine = np.array([[-1, 0, 0, 0],  # nopep8
                               [0, 0, 1, 0],  # nopep8
                               [0, -1, 0, 0],  # nopep8
                               [0, 0, 0, 1]], dtype=float)  # nopep8
            pcrs = np.append(np.array(array.shape[:3]) / 2, 1)
            affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        nib.save(nib.Nifti1Image(array, affine), filename)
    elif filename.endswith('.npz'):
        np.savez_compressed(filename, vol=array)
    else:
        raise ValueError('unknown filetype for %s' % filename)


def load_labels(arg):
    """
    Load label maps and return a list of unique labels as well as all maps.

    Parameters:
        arg: Path to folder containing label maps, string for globbing, or a list of these.

    Returns:
        np.array: List of unique labels.
        list: List of label maps, each as a np.array.
    """
    if not isinstance(arg, (tuple, list)):
        arg = [arg]

    # List files.
    import glob
    ext = ('.nii.gz', '.nii', '.mgz', '.npy', '.npz')
    files = [os.path.join(f, '*') if os.path.isdir(f) else f for f in arg]
    files = sum((glob.glob(f) for f in files), [])
    files = [f for f in files if f.endswith(ext)]

    # Load labels.
    if len(files) == 0:
        raise ValueError(f'no labels found for argument "{files}"')
    label_maps = []
    shape = None
    for f in files:
        x = np.squeeze(load_volfile(f))
        if shape is None:
            shape = np.shape(x)
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError(f'file "{f}" has non-integral data type')
        if not np.all(x.shape == shape):
            raise ValueError(f'shape {x.shape} of file "{f}" is not {shape}')
        label_maps.append(x)

    return np.unique(label_maps), label_maps
    
def pad(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded, slices

def crop(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """

    if array.shape[:-1] == tuple(shape):
        return array, ...

    #croped = np.zeros(shape+[1,], dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(array.shape, shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, shape)])
    croped = array[slices]

    return croped


def resize(array, factor, batch_axis=False):
    """
    Resizes an array by a given factor. This expects the input array to include a feature dimension.
    Use batch_axis=True to avoid resizing the first (batch) dimension.
    """

    if factor == 1:
        return array
    else:

        if not batch_axis:
            if isinstance(factor, tuple):
                dim_factors = list(factor) + [1]
            else:
                dim_factors = [factor for _ in array.shape[:-1]] + [1]
        else:
            if isinstance(factor, tuple):
                dim_factors = [1]+list(factor) + [1]
            else:
                dim_factors = [1] + [factor for _ in array.shape[1:-1]] + [1]
        return scipy.ndimage.interpolation.zoom(array, dim_factors, order=0)

