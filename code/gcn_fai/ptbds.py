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
tfm_norm = Normalize(*stats, TfmType.NO)

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
        np_file = np.load(self.fnames[idx])
        if pidx is None:
            pidx = np.random.randint(np_file.shape[0])
#         pdb.set_trace()
        return np_file[pidx]

    def get_y(self, idx):
        return self.labels[idx]
    
    
class PTBModelData(ImageData):
    @classmethod
    def from_dataset(cls, tdir, tmp_tdir, trn_csv, val_csv, test_csv, bs, num_workers, tfms=None):
        trn_ds = PTBData(tdir, trn_csv, tfms)
        val_ds = PTBData(tdir, val_csv, tfms)
        fix_ds = PTBData(tdir, trn_csv, tfms)
        aug_ds = PTBData(tdir, val_csv, tfms)
        test_ds = PTBData(tdir, test_csv, tfms)
        test_aug_ds = PTBData(tdir, test_csv, tfms)
        res = [trn_ds, val_ds, fix_ds, aug_ds, test_ds, test_aug_ds]
        return cls(tmp_tdir, res, bs, num_workers, classes=trn_ds.classes)
    
trn_csv = PATH / 'train_only_labels.csv'
val_csv = PATH / 'val_only_labels.csv'
test_csv = PATH / 'test_only_labels.csv'

data = PTBModelData.from_dataset(BPATH, PATH, trn_csv, val_csv, test_csv, bs=64, num_workers=4, tfms=tfm_norm)