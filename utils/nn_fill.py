import sys

sys.path.append('../')

import numpy as np
# import matplotlib.pyplot as plt
from utils.nnfc.nnfc import _nn_fill_c

import time


def NN_fill(image, depth, mask=None, mode="Nearest"):
    '''
    Nearest depth fill using pattern_mask.

    Args:
        image: RGB image
        depth: dense depth map (should be the same size as RGB image)
        mask: sparse points (if None, it will be the non-zero points of depth)
        mode: 'Nearest' for fill the zero points using nearest search,'Sparse' for compare with all points in mask pattren

    Returns:
        NN fill map `s1` and euclidean distance transforms `s2`
    '''
    assert image.shape[0:2] == depth.shape[0:2]
    assert mode == "Nearest" or mode == "Sparse"

    if mask is None:
        mask = depth > 0

    depth = np.array(depth, dtype=np.float)
    mask = np.array(mask, dtype=np.uint8)

    s_mode = 0
    if mode == "Nearest":
        s_mode = 0
    elif mode == "Sparse":
        s_mode = 1

    s1, s2 = _nn_fill_c(depth, mask, s_mode)

    return s1, s2


def generate_mask(Ah, Aw, H, W):
    """
    Modified from: https://github.com/kvmanohar22/sparse_depth_sensing/blob/master/utils/utils.py
    Generates mask for the depth data Sparsity is: depth_values
    Args:
        Ah: Downsampling factor along h
        Aw: Downsampling factor along w
        H : Image height
        W : Image width
    Returns:
        binary mask of dimensions (H, W) where 1 equals
        denotes actual ground truth data is retained
    """
    mask = np.zeros((H, W))
    mask[0:None:Ah, 0:None:Aw] = 1
    return mask


def sparse_inputs(img, depth, mask, Ah, Aw):
    """
    From: https://github.com/kvmanohar22/sparse_depth_sensing/blob/master/utils/utils.py

    This implementation is memory inefficient.

    Generate sparse inputs
    Args:
        img  : input image
        depth: ground truth depth map for the given image
        mask : pixels at which the ground truth depth is to be retained
        Ah: Downsampling factor along the height
        Aw: Downsampling factor along the width
    Returns:
        sparse inputs
    """
    H, W, _ = img.shape
    valid_depth_points = depth > 0
    S1 = np.zeros((H, W), dtype=np.float32)
    S2 = np.zeros((H, W), dtype=np.float32)

    # Make sure the sampling points have valid depth
    dist_tuple = [(i, j) for i in range(H) for j in range(W) if valid_depth_points[i, j]]
    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1 and depth[i, j] == 0:
                print(i, j)
                dist_transform = [np.sqrt(
                    np.square(i - vec[0]) +
                    np.square(j - vec[1]))
                    for vec in dist_tuple if vec != [i, j]
                ]
                closest_pixel = np.argmin(dist_transform)
                mask[i, j] = 0
                x, y = dist_tuple[closest_pixel]
                mask[x, y] = 1

    Sh = [i for i in range(H) for j in range(W) if mask[i, j]]
    Sw = [j for i in range(H) for j in range(W) if mask[i, j]]
    idx_to_depth = {}
    for i, (x, y) in enumerate(zip(Sh, Sw)):
        idx_to_depth[i] = x * W + y

    Sh, Sw = np.array(Sh), np.array(Sw)
    Hd, Wd = np.empty((H, W)), np.empty((H, W))
    Hd.T[:, ] = np.arange(H)
    Wd[:, ] = np.arange(W)
    Hd, Wd = Hd[..., None], Wd[..., None]
    Hd2 = np.square(Hd - Sh)  # This two broadcast will consume lots of memory
    Wd2 = np.square(Wd - Sw)  # when the number of sparse points and image size a little bit larger
    dmap = np.sqrt(Hd2 + Wd2)
    dmap_arg = np.argmin(dmap, axis=-1)
    dmap_arg = dmap_arg.ravel()
    dmap_arg = np.array([idx_to_depth[i] for i in dmap_arg])
    S1 = depth.ravel()[dmap_arg].reshape(H, W)[None]
    S2 = np.sqrt(np.min(dmap, axis=-1))[None]
    return np.concatenate((S1, S2), axis=0)


if __name__ == "__main__":

    # test_dataset = 'kitti'
    test_dataset = 'nyuv2'

    if test_dataset == 'kitti':
        test_dir = '../kitti_test_imgs'
    elif test_dataset == 'nyuv2':
        test_dir = '../nyuv2_test_imgs'
    else:
        exit(-1)

    for i in range(10):
        image = np.load(test_dir + '/image{}.npy'.format(i))
        depth = np.load(test_dir + '/depth{}.npy'.format(i))

        # if test_dataset == 'nyuv2':
        mask = generate_mask(24, 24, image.shape[0], image.shape[1])
        _mask = mask.copy()

        t1 = time.time()
        if test_dataset == 'kitti':
            s1, s2 = NN_fill(image, depth, mask)
        else:
            s1, s2 = NN_fill(image, depth, mask)
        t2 = time.time()
        print('Spend {}'.format(t2 - t1))

        # plt.subplot(221)
        # plt.imshow(image)
        # plt.title('RGB')
        # plt.axis('off')
        #
        # plt.subplot(222)
        # plt.imshow(depth)
        # plt.title('Depth')
        # plt.axis('off')
        #
        # plt.subplot(223)
        # plt.imshow(s1)
        # plt.title('S1')
        # plt.axis('off')
        #
        # plt.subplot(224)
        # plt.imshow(s2)
        # plt.title('S2')
        # plt.axis('off')
        #
        # plt.show()
