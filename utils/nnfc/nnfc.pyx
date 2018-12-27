'''
nn fill cython code
'''
# cython: infer_types=True

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.uint8_t uint8

DTYPE = np.float


@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing
def _nn_fill_c(double[:,::1] depth, uint8[:,::1] mask, int search_mode):
    '''
    Nearest depth fill using pattern_mask.

    Args:
        depth: dense depth map
        mask: sparse points (should be the same size as depth)
        mode: 0 for fill the zero points using nearest search,1 for compare with all points in mask pattren

    Returns:
        NN fill map `s1` and euclidean distance transforms `s2`
    '''
    assert tuple(depth.shape) == tuple(mask.shape)
    assert search_mode == 0 or search_mode == 1

    cdef int h, w

    h = depth.shape[0]
    w = depth.shape[1]

    s1 = np.zeros((h,w), dtype=DTYPE)
    s2 = np.zeros((h,w), dtype=DTYPE)
    cdef double[:, ::1] s1_view = s1
    cdef double[:, ::1] s2_view = s2

    # define variables
    cdef double dis2, val, min_dis2, min_val, MAX_DIS2
    cdef int range_ii, range_jj, srch_i, srch_j,new_i,new_j
    cdef int in_image_flag,i,j,ii,jj

    MAX_DIS2 = h*h+w*w+1

    # the first for loop is the adjust of the mask
    for i in range(h):  # row
        for j in range(w):  # col
            if mask[i,j] == 1 and depth[i,j] == 0:
                # search...
                new_i, new_j = 0, 0
                range_ii, range_jj = 1, 1
                while min_dis2 == MAX_DIS2:
                    # in_image_flag = 0
                    # nearnst search
                    ii = -range_ii
                    for jj in range(-range_jj, range_jj + 1):
                        # in_image_flag = 0
                        # image range
                        srch_i, srch_j = i + ii, j + jj
                        if srch_i >= 0 and srch_i < h and srch_j >= 0 and srch_j < w:
                            # in_image_flag = 1
                            if depth[srch_i, srch_j] != 0:
                                dis2 = ii * ii + jj * jj
                                if dis2 < min_dis2:
                                    min_dis2 = dis2
                                    new_i = srch_i
                                    new_j = srch_j
                    ii = range_ii
                    for jj in range(-range_jj, range_jj + 1):
                        # in_image_flag = 0
                        # image range
                        srch_i, srch_j = i + ii, j + jj
                        if srch_i >= 0 and srch_i < h and srch_j >= 0 and srch_j < w:
                            # in_image_flag = 1
                            if depth[srch_i, srch_j] != 0:
                                dis2 = ii * ii + jj * jj
                                if dis2 < min_dis2:
                                    min_dis2 = dis2
                                    new_i = srch_i
                                    new_j = srch_j
                    jj = -range_jj
                    for ii in range(-range_ii+1, range_ii):
                        # in_image_flag = 0
                        # image range
                        srch_i, srch_j = i + ii, j + jj
                        if srch_i >= 0 and srch_i < h and srch_j >= 0 and srch_j < w:
                            # in_image_flag = 1
                            if depth[srch_i, srch_j] != 0:
                                dis2 = ii * ii + jj * jj
                                if dis2 < min_dis2:
                                    min_dis2 = dis2
                                    new_i = srch_i
                                    new_j = srch_j
                    jj = range_jj
                    for ii in range(-range_ii+1, range_ii):
                        # in_image_flag = 0
                        # image range
                        srch_i, srch_j = i + ii, j + jj
                        if srch_i >= 0 and srch_i < h and srch_j >= 0 and srch_j < w:
                            # in_image_flag = 1
                            if depth[srch_i, srch_j] != 0:
                                dis2 = ii * ii + jj * jj
                                if dis2 < min_dis2:
                                    min_dis2 = dis2
                                    new_i = srch_i
                                    new_j = srch_j

                    # if in_image_flag == 0:
                    #     break
                    range_ii += 1
                    range_jj += 1

                mask[i,j]=0
                mask[new_i,new_j]=1

    if search_mode == 1:
        mask_ij_array = np.argwhere(mask==np.uint8(1))
        mask_list = []
        for idx in range(mask_ij_array.shape[0]):
            i = mask_ij_array[idx,0]
            j = mask_ij_array[idx,1]
            mask_list.append((i,j))
        # mask_list = [(i,j) for i in range(h) for j in range(w) if mask[i,j]==np.uint8(1)]

    for i in range(h):  # row
        for j in range(w):  # col
            if mask[i,j] == 0:
                # search...
                min_dis2, min_val = MAX_DIS2, 0
                range_ii, range_jj = 1, 1
                while min_dis2 == MAX_DIS2:
                    # in_image_flag = 0

                    # nearnst search
                    if search_mode == 0:
                        ii = -range_ii
                        for jj in range(-range_jj, range_jj + 1):
                            # in_image_flag = 0
                            # image range
                            srch_i, srch_j = i + ii, j + jj
                            if srch_i >= 0 and srch_i < h and srch_j >= 0 and srch_j < w:
                                # in_image_flag = 1
                                if mask[srch_i, srch_j] == 1:
                                    dis2 = ii * ii + jj * jj
                                    val = depth[srch_i, srch_j]
                                    if dis2 < min_dis2:
                                        min_dis2 = dis2
                                        min_val = val
                        ii = range_ii
                        for jj in range(-range_jj, range_jj + 1):
                            # in_image_flag = 0
                            # image range
                            srch_i, srch_j = i + ii, j + jj
                            if srch_i >= 0 and srch_i < h and srch_j >= 0 and srch_j < w:
                                # in_image_flag = 1
                                if mask[srch_i, srch_j] == 1:
                                    dis2 = ii * ii + jj * jj
                                    val = depth[srch_i, srch_j]
                                    if dis2 < min_dis2:
                                        min_dis2 = dis2
                                        min_val = val
                        jj = -range_jj
                        for ii in range(-range_ii+1, range_ii):
                            # in_image_flag = 0
                            # image range
                            srch_i, srch_j = i + ii, j + jj
                            if srch_i >= 0 and srch_i < h and srch_j >= 0 and srch_j < w:
                                # in_image_flag = 1
                                if mask[srch_i, srch_j] == 1:
                                    dis2 = ii * ii + jj * jj
                                    val = depth[srch_i, srch_j]
                                    if dis2 < min_dis2:
                                        min_dis2 = dis2
                                        min_val = val
                        jj = range_jj
                        for ii in range(-range_ii+1, range_ii):
                            # in_image_flag = 0
                            # image range
                            srch_i, srch_j = i + ii, j + jj
                            if srch_i >= 0 and srch_i < h and srch_j >= 0 and srch_j < w:
                                # in_image_flag = 1
                                if mask[srch_i, srch_j] == 1:
                                    dis2 = ii * ii + jj * jj
                                    val = depth[srch_i, srch_j]
                                    if dis2 < min_dis2:
                                        min_dis2 = dis2
                                        min_val = val

                    # compare the empty points with all mask points
                    elif search_mode == 1:
                        for ii,jj in mask_list:
                            if mask[ii,jj] == 1 and depth[ii,jj] != 0:
                                dis2 = (ii-i)*(ii-i)+(jj-j)*(jj-j)
                                val = depth[ii,jj]

                                if dis2 < min_dis2:
                                    min_dis2 = dis2
                                    min_val = val

                    # if in_image_flag == 0:
                    #     break
                    range_ii += 1
                    range_jj += 1

                s1_view[i, j] = min_val
                s2_view[i, j] = np.sqrt(min_dis2)
            else:
                s1_view[i, j] = depth[i, j]
                s2_view[i, j] = 0


    return s1, s2
