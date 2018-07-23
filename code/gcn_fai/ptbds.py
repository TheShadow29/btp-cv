from all_imports import *

PATH = Path('/scratch/arka/Ark_git_files/ecg_data/')
BPATH = PATH / 'beats_data'

def get_stats(tr_fnames):
    l1 = []
    for f in tqdm(tr_fnames):
        l1.append(np.load(f)[0])
    l1np = np.array(l1)
    m1 = l1np.mean(axis=0).mean(axis=0)
    l2np = np.square(l1np)
    m2 = l2np.mean(axis=0).mean(axis=0)
    return m1, np.sqrt(m2 - np.square(m1))

stats = (np.array([-0.00045, -0.00065,  0.0002 ,  0.00053, -0.0004 , -0.00028,  0.00134,  0.00166,  0.00129,  0.00045,
        -0.00055, -0.00075, -0.00051, -0.00046, -0.00091]),
 np.array([0.07205, 0.07793, 0.06985, 0.06643, 0.05931, 0.06463, 0.10812, 0.16106, 0.15547, 0.14078, 0.12046,
        0.09665, 0.09572, 0.06398, 0.08093]))
# tfm_norm = Normalize(*stats, TfmType.NO)



class PTBData(BaseDataset):
    def __init__(self, tdir, csv_file, transform=None):
        self.tdir = tdir
        self.csv_file = csv_file
        self.fnames, self.labels, self.classes = csv_source(self.tdir, self.csv_file, suffix='_beats.npy')
        super().__init__(transform=transform)

    
    def get_n(self):
        return len(self.fnames)
    
    def get_c(self):
        return len(self.classes)
    
    def get_sz(self):
        return 149
    
    def get_x(self, idx, pidx=None):
        while idx==136:
            idx = np.random.randint(len(self.fnames))
        np_file = np.load(self.fnames[idx])

        if pidx is None:
            pidx = np.random.randint(np_file.shape[0])
#         pdb.set_trace()
        return np_file[pidx]

    def get_y(self, idx):
        return self.labels[idx]
    
    
class PTBValData(PTBData):
    def __init__(self, tdir, csv_file, transform=None, num_patches=50):
        self.tdir = tdir
        self.csv_file = csv_file
        self.fnames, self.labels, self.classes = csv_source(self.tdir, self.csv_file, suffix='_beats.npy')
        self.num_patches = num_patches
        super().__init__(tdir=tdir, csv_file=csv_file, transform=transform)
    
    def get_n(self):
        return len(self.fnames) * self.num_patches
    
    def get1item(self, idx):
        act_idx = idx // self.num_patches
        pidx = idx % self.num_patches
        x,y = self.get_x(act_idx, pidx),self.get_y(act_idx)
        return self.get(self.transform, x, y)


    
class PTBModelData(ImageData):
    @classmethod
    def from_dataset(cls, tdir, tmp_tdir, trn_csv, val_csv, test_csv, bs, num_workers, tfms=None):
        trn_ds = PTBValData(tdir, trn_csv, tfms[0], num_patches=10)
        val_ds = PTBValData(tdir, val_csv, tfms[1])
        fix_ds = PTBValData(tdir, trn_csv, tfms[1])
        aug_ds = PTBValData(tdir, val_csv, tfms[0])
        test_ds = PTBValData(tdir, test_csv, tfms[1])
        test_aug_ds = PTBValData(tdir, test_csv, tfms[0])
        res = [trn_ds, val_ds, fix_ds, aug_ds, test_ds, test_aug_ds]
        return cls(tmp_tdir, res, bs, num_workers, classes=trn_ds.classes)
    
trn_csv = PATH / 'train_only_two_class_labels_id_gami.csv'
val_csv = PATH / 'val_only_two_class_labels_id_gami.csv'
test_csv = PATH / 'test_only_two_class_labels_id_gami.csv'


class ChannelOrder1d():
    '''
    changes image array shape from (h, w, 3) to (3, h, w). 
    tfm_y decides the transformation done to the y element. 
    '''
    def __init__(self, tfm_y=TfmType.NO): self.tfm_y=tfm_y

    def __call__(self, x, y):
        x = np.rollaxis(x, 1)
        #if isinstance(y,np.ndarray) and (len(y.shape)==3):
        if self.tfm_y==TfmType.PIXEL: y = np.rollaxis(y, 2)
        elif self.tfm_y==TfmType.CLASS: y = y[...,0]
        return x,y
    
class Transforms1d():
    def __init__(self, sz, tfms, normalizer, denorm, crop_type=CropType.CENTER,
                 tfm_y=TfmType.NO, sz_y=None):
        if sz_y is None: sz_y = sz
        self.sz,self.denorm,self.norm,self.sz_y = sz,denorm,normalizer,sz_y
        self.tfms = tfms
        if normalizer is not None: self.tfms.append(normalizer)
        self.tfms.append(ChannelOrder1d(tfm_y))

    def __call__(self, im, y=None): return compose(im, y, self.tfms)
    def __repr__(self): return str(self.tfms)

def ecg_gen(normalizer, denorm, sz, tfms=None, max_zoom=None, pad=0, crop_type=None,
              tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT, scale=None):
    """
    Generate a standard set of transformations
    Arguments
    ---------
     normalizer :
         image normalizing function
     denorm :
         image denormalizing function
     sz :
         size, sz_y = sz if not specified.
     tfms :
         iterable collection of transformation functions
     max_zoom : float,
         maximum zoom
     pad : int,
         padding on top, left, right and bottom
     crop_type :
         crop type
     tfm_y :
         y axis specific transformations
     sz_y :
         y size, height
     pad_mode :
         cv2 padding style: repeat, reflect, etc.
    Returns
    -------
     type : ``Transforms``
         transformer for specified image operations.
    See Also
    --------
     Transforms: the transformer object returned by this function
    """
    if tfm_y is None: tfm_y=TfmType.NO
    if tfms is None: tfms=[]
    elif not isinstance(tfms, collections.Iterable): tfms=[tfms]
    if sz_y is None: sz_y = sz

    return Transforms1d(sz, tfms, normalizer, denorm, crop_type,
                      tfm_y=tfm_y, sz_y=sz_y)

def tfms_from_stats1d(stats, sz, aug_tfms=None, max_zoom=None, pad=0, crop_type=CropType.RANDOM,
                    tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT, norm_y=True, scale=None):
    """ Given the statistics of the training image sets, returns separate training and validation transform functions
    """
    tfm_norm = Normalize(*stats, tfm_y=TfmType.NO)
    tfm_denorm = Denormalize(*stats)
    val_crop = CropType.CENTER if crop_type in (CropType.RANDOM,CropType.GOOGLENET) else crop_type
    val_tfm = ecg_gen(tfm_norm, tfm_denorm, sz, pad=pad, crop_type=val_crop,
            tfm_y=tfm_y, sz_y=sz_y, scale=scale)
    trn_tfm = ecg_gen(tfm_norm, tfm_denorm, sz, pad=pad, crop_type=crop_type,
            tfm_y=tfm_y, sz_y=sz_y, tfms=aug_tfms, max_zoom=max_zoom, pad_mode=pad_mode, scale=scale)
    return trn_tfm, val_tfm

tfms = tfms_from_stats1d(stats, 149)
# tfm_norm = Normalize(*stats, TfmType.NO)
# trn_tfm = Transforms1d(sz, tfm, normalizer, denorm, crop_type,
#                       tfm_y=tfm_y, sz_y=sz_y)

data = PTBModelData.from_dataset(BPATH, PATH, trn_csv, val_csv, test_csv, bs=16, num_workers=4, tfms=tfms)
data2 = PTBModelData.from_dataset(BPATH, PATH, trn_csv, test_csv, val_csv, bs=16, num_workers=4, tfms=tfms)