import sys

sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from utils.nnfc.nnfc import _nn_fill_c


def NN_fill(image, depth, pattern_mask=None, mode="Nearest"):
    '''
    Nearest depth fill using pattern_mask.

    Args:
        image: RGB image
        depth: dense depth map (should be the same size as RGB image)
        pattern_mask: sparse points (if None, it will be the non-zero points of depth)
        mode: 'Nearest' for fill the zero points using nearest search,'Sparse' for compare with all points in mask pattren

    Returns:
        NN fill map `s1` and euclidean distance transforms `s2`
    '''
    assert image.shape[0:2] == depth.shape[0:2]
    assert mode == "Nearest" or mode == "Sparse"

    if pattern_mask == None:
        pattern_mask = np.where(depth > 0, True, False)

    depth = np.array(depth, dtype=np.float)
    pattern_mask = np.array(pattern_mask, dtype=np.uint8)

    s_mode = 0
    if mode == "Nearest":
        s_mode = 0
    elif mode == "Sparse":
        s_mode = 1

    s1, s2 = _nn_fill_c(depth, pattern_mask, s_mode)

    return s1, s2


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

    test_dir = '/mnt/dDocuments/PycharmProjects/prj/DensefromRGBS/kitti_test_imgs'
    for i in range(10):
        image = np.load(test_dir + '/image{}.npy'.format(i))
        depth = np.load(test_dir + '/depth{}.npy'.format(i))
        ptsim = np.load(test_dir + '/ptsim{}.npy'.format(i))

        # mask = np.where(depth > 0, True, False)
        # s1, s2 = sparse_inputs(image, depth, mask, 1, 1)

        s1, s2 = NN_fill(image, depth)

        max_depth = np.max(depth)
        depth_show = np.copy(depth)
        depth_show = np.where(depth_show == 0, max_depth, depth_show)
        depth_show = max_depth - depth_show

        plt.subplot(221)
        plt.imshow(image)
        plt.subplot(222)
        plt.imshow(depth_show)
        plt.subplot(223)
        plt.imshow(s1)
        plt.subplot(224)
        plt.imshow(s2)
        plt.show()
