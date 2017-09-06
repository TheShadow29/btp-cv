import scipy.io as sio
import skimage.io as skio


class helper_mat_file(object):
    def __init__(self, mat_file, img_file):
        self.img = img_file
        self.seg_img = mat_file['segImgI']
        self.adj_graph = mat_file['graphI']
        self.sI = mat_file['sI']
        self.bg_prob = mat_file['bgProb']
        self.segments = mat_file['segmentsI']
        return


if __name__ == '__main__':
    print('Starting to Load Mat files')

    mat_file = sio.loadmat(
        '../../data/image_coseg_avik_data/weizmann_horse_db/seg200/horse001.mat')
    img_file = skio.imread('../../data/image_coseg_avik_data/weizmann_horse_db/rgb/horse001.jpg')
    h = helper_mat_file(mat_file, img_file)
    # skio.imshow(img_file)
