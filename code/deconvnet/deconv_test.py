import sys
sys.path.append('/home/arktheshadow/Caffe/deconv_caffe/caffe/python/')
import caffe
import caffe_io as cio
# from caffe import Net
# sys.path.pop()
# sys.path.append('/home/arktheshadow/Caffe/caffe/python/')
# import caffe
import numpy as np
import skimage.io as skio


def fp_one_img(img_str, net, transformer):
    im = cio.load_image(img_str)
    net.blobs['data'].reshape(1, 3, 224, 224)
    net.blobs['data'].data[:, :, :] = transformer.preprocess('data', im)

    out = net.forward()
    return net


def show_img(img_shuffled, ind):
    if ind == -1:
        a1 = img_shuffled[0, :, :, :]
        a1_shape = a1.shape
        a2 = list()
        for i in range(a1_shape[0]):
            a2.append(a1[ind, :, :])
        # skio.imshow_collection(a2)
        # skio.show()
        # fig.show()
        return a2
    else:
        a1 = img_shuffled[0, ind, :, :]
        skio.imshow(a1)
        skio.show()
        return a1


descriptor_path = '../../data/caffe_models/deconvnet/DeconvNet_inference_deploy.prototxt'
weights_path = '../../data/caffe_models/deconvnet/DeconvNet_trainval_inference.caffemodel'
ilsvrc_mean_path = '/home/arktheshadow/Caffe/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'

net = caffe.Net(descriptor_path, weights_path)

transformer = cio.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(ilsvrc_mean_path).mean(1).mean(1))
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_raw_scale('data', 255.0)

img_str = '../../data/video_cosegmentation/ObMiC_dataset/data/\
Skating_All/Skating_2/skating-4586.jpg'
net1 = fp_one_img(img_str, net, transformer)
# skio.imshow()
show_img(net1.blobs['seg-score'].data, 0)
