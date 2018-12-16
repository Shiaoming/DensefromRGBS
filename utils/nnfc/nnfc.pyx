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
def _nn_fill_c(uint8[:,:,::1] image, double[:,::1] depth, uint8[:,::1] pattern_mask):
    '''
    :param image:
    :param depth:
    :param pattern_mask:
    :return:
    '''
    cdef Py_ssize_t h, w

    h = depth.shape[0]
    w = depth.shape[1]

    s1 = np.zeros((h,w), dtype=DTYPE)
    s2 = np.zeros((h,w), dtype=DTYPE)
    cdef double[:, ::1] s1_view = s1
    cdef double[:, ::1] s2_view = s2

    # TODO: the first for loop is the adjust of the mask
    # for ....

    # define variables
    cdef float dis2, val, min_dis2, min_val, MAX_DIS2
    cdef Py_ssize_t range_ii, range_jj, srch_i, srch_j
    cdef int in_image_flag,ii,jj

    MAX_DIS2 = h*h+w*w+1

    for i in range(h):  # row
        for j in range(w):  # col
            if pattern_mask[i,j] == 0 and depth[i,j] == 0:
                # search...
                min_dis2, min_val = MAX_DIS2, 0
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
                            if pattern_mask[srch_i, srch_j] == 1:
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
                            if pattern_mask[srch_i, srch_j] == 1:
                                dis2 = ii * ii + jj * jj
                                val = depth[srch_i, srch_j]
                                if dis2 < min_dis2:
                                    min_dis2 = dis2
                                    min_val = val
                    jj = -range_jj
                    for ii in range(-range_ii, range_ii + 1):
                        # in_image_flag = 0
                        # image range
                        srch_i, srch_j = i + ii, j + jj
                        if srch_i >= 0 and srch_i < h and srch_j >= 0 and srch_j < w:
                            # in_image_flag = 1
                            if pattern_mask[srch_i, srch_j] == 1:
                                dis2 = ii * ii + jj * jj
                                val = depth[srch_i, srch_j]
                                if dis2 < min_dis2:
                                    min_dis2 = dis2
                                    min_val = val
                    jj = -range_jj
                    for ii in range(-range_ii, range_ii + 1):
                        # in_image_flag = 0
                        # image range
                        srch_i, srch_j = i + ii, j + jj
                        if srch_i >= 0 and srch_i < h and srch_j >= 0 and srch_j < w:
                            # in_image_flag = 1
                            if pattern_mask[srch_i, srch_j] == 1:
                                dis2 = ii * ii + jj * jj
                                val = depth[srch_i, srch_j]
                                if dis2 < min_dis2:
                                    min_dis2 = dis2
                                    min_val = val

                    # compare the empty points with all mask points
                    # for ii in range(h):
                    #     for jj in range(w):
                    #         if pattern_mask[ii,jj] == 1 and depth[ii,jj] != 0:
                    #             dis2 = (ii-i)*(ii-i)+(jj-j)*(jj-j)
                    #             val = depth[ii,jj]
                    #
                    #             if dis2 < min_dis2:
                    #                 min_dis2 = dis2
                    #                 min_val = val

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