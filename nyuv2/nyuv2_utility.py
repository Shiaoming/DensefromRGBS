'''
Python version of nyu toolbox rgb and depth image undistort, depth image projection to rgb image plane.
'''
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from utils.viz_utility import pc_viewer

# The maximum depth used, in meters.
maxDepth = 10

# RGB Intrinsic Parameters
fx_rgb = 5.1885790117450188e+02
fy_rgb = 5.1946961112127485e+02
cx_rgb = 3.2558244941119034e+02
cy_rgb = 2.5373616633400465e+02

K_rgb = np.array([[fx_rgb, 0, cx_rgb],
                  [0, fy_rgb, cy_rgb],
                  [0, 0, 1]])
Krgbinv = np.linalg.inv(K_rgb)

# RGB Distortion Parameters
k1_rgb = 2.0796615318809061e-01
k2_rgb = -5.8613825163911781e-01
p1_rgb = 7.2231363135888329e-04
p2_rgb = 1.0479627195765181e-03
k3_rgb = 4.9856986684705107e-01

kc_rgb = np.array([k1_rgb, k2_rgb, p1_rgb, p2_rgb, k3_rgb])

# Depth Intrinsic Parameters
fx_d = 5.8262448167737955e+02
fy_d = 5.8269103270988637e+02
cx_d = 3.1304475870804731e+02
cy_d = 2.3844389626620386e+02

K_d = np.array([[fx_d, 0, cx_d],
                [0, fy_d, cy_d],
                [0, 0, 1]])
Kdinv = np.linalg.inv(K_d)

# Depth Distortion Parameters
k1_d = -9.9897236553084481e-02
k2_d = 3.9065324602765344e-01
p1_d = 1.9290592870229277e-03
p2_d = -1.9422022475975055e-03
k3_d = -5.1031725053400578e-01

kc_d = np.array([k1_d, k2_d, p1_d, p2_d, k3_d])

# Rotation
R = np.array([[9.9997798940829263e-01, 5.0518419386157446e-03, 4.3011152014118693e-03],
              [-5.0359919480810989e-03, 9.9998051861143999e-01, -3.6879781309514218e-03],
              [-4.3196624923060242e-03, 3.6662365748484798e-03, 9.9998394948385538e-01]])
R = np.linalg.inv(R)

# 3D Translation
t_x = 2.5031875059141302e-02
t_z = -2.9342312935846411e-04
t_y = 6.6238747008330102e-04
T = -np.array([t_x, t_z, t_y])

# Parameters for making depth absolute.
depthParam1 = 351.3
depthParam2 = 1092.5

H, W = 480, 640
u, v = np.meshgrid(np.arange(W), np.arange(H))
ones = np.ones((H, W))

uvone = np.stack((u, v, ones), axis=2)  # H*W*3
uvone = uvone.reshape((-1, 3))  # (H*W)*3

PRJd = (Kdinv @ (uvone.T)).T.reshape((H, W, 3))  # K^-1*uvone, size: H*W*3


def depth_rel2depth_abs(imgDepthOrig):
    imgDepthAbs = depthParam1 / (depthParam2 - imgDepthOrig)
    imgDepthAbs[imgDepthAbs > maxDepth] = maxDepth
    imgDepthAbs[imgDepthAbs < 0] = 0
    return imgDepthAbs


def depth_plane2depth_world(imgDepthAbs):
    global PRJd

    imgDepthAbs = imgDepthAbs[:, :, np.newaxis]
    xyz = PRJd * imgDepthAbs

    return xyz.reshape((-1, 3))

    # global u, v
    #
    # X = (u - cx_d) * imgDepthAbs / fx_d
    # Y = (v - cy_d) * imgDepthAbs / fy_d
    # Z = imgDepthAbs
    # return np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)


def depth_world2rgb_world(points3d):
    global R, T
    T = T.reshape((3, 1))
    points3d = R @ points3d.T + T
    return points3d.T


def depth_world2depth_plane(points3d):
    global K_d

    prjuvz = (K_d @ points3d.T).T  # K @ (x,y,z)
    prjuvz[:, 0:2] = prjuvz[:, 0:2] / prjuvz[:, -1:]

    return prjuvz

    # X_world = points3d[:, 0]
    # Y_world = points3d[:, 1]
    # Z_world = points3d[:, 2]
    #
    # X_plane = (X_world * fx_d / Z_world) + cx_d
    # Y_plane = (Y_world * fy_d / Z_world) + cy_d
    #
    # return np.stack((X_plane.ravel(), Y_plane.ravel(), points3d[:,2].ravel()), axis=1)


def rgb_world2rgb_plane(points3d):
    global K_rgb

    prjuvz = (K_rgb @ points3d.T).T  # K @ (x,y,z)
    prjuvz[:, 0:2] = prjuvz[:, 0:2] / prjuvz[:, -1:]

    return prjuvz

    # X_world = points3d[:, 0]
    # Y_world = points3d[:, 1]
    # Z_world = points3d[:, 2]
    #
    # X_plane = (X_world * fx_rgb / Z_world) + cx_rgb
    # Y_plane = (Y_world * fy_rgb / Z_world) + cy_rgb
    #
    # return np.stack((X_plane.ravel(), Y_plane.ravel(), points3d[:,2].ravel()), axis=1)


def project_depth_map(rgb, imgDepth):
    """
    project_depth_map(rgb, imgDepth)

        undistort rgb and depth image, project depth image to rgb image plane.

        Parameters
        ----------
        rgb: numpy array
            raw rgb image read from '.ppm' file

        imgDepth: numpy array
            raw depth image read from '.pgm' file

        -------
        Returns
         rgbUndistort: undistorted rgb image (uint8)

         depthOriginal: undistorted depth image (float)

         depthOut: undistorted and projected depth image (float)

    """
    def sub2ind(matrixSize, rowSub, colSub):
        m, n = matrixSize
        return rowSub * (n - 1) + colSub - 1

    imgDepth = imgDepth.astype(np.uint16)
    imgDepth = imgDepth.byteswap(inplace=True)

    h, w = imgDepth.shape
    # Undistort the rgb image.
    rgbUndistort = cv2.undistort(rgb, K_rgb, kc_rgb)

    noiseMask = 255 * (imgDepth == np.max(imgDepth)).astype(np.uint8)
    # Undistort the noise mask.
    noiseMask = cv2.undistort(noiseMask, K_d, kc_d)
    noiseMask = noiseMask > 0

    # Undistort the depth.
    imgDepth = imgDepth.astype(np.float)
    imgDepth = cv2.undistort(imgDepth, K_d, kc_d)

    # Fix issues introduced by distortion.
    imgDepth[imgDepth < 600] = 2047
    imgDepth[noiseMask] = 2047

    depthOriginal = depth_rel2depth_abs(imgDepth)
    points3d = depth_plane2depth_world(depthOriginal)
    # prjuvz = depth_world2depth_plane(points3d)
    points3d = depth_world2rgb_world(points3d)

    prjuvz = rgb_world2rgb_plane(points3d)

    # pc_viewer(points3d, mode='point')

    # Finally, project back onto the RGB plane
    v = np.round(prjuvz[:, 1])
    u = np.round(prjuvz[:, 0])

    goodmask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    good_uvz = prjuvz[goodmask, :]

    # the order from far to near, using depth order is a more efficient way
    # to project to image plane when depth is dense. In kitti dataset, the depth is sparse,
    # the duplicated way is more faster, while in nyv2 dataset, the order way is faster
    order = np.argsort(-good_uvz[:, 2])
    # project to image
    depthOut = np.zeros((h, w)).astype(np.float32)
    for idx in range(order.shape[0]):
        v = good_uvz[order[idx], 1].astype(np.int)
        u = good_uvz[order[idx], 0].astype(np.int)
        depthOut[v, u] = good_uvz[order[idx], 2]

    # depthOut[good_uvz[:, 1].astype(np.int), good_uvz[:, 0].astype(np.int)] = good_uvz[:, 2]
    #
    # # find the duplicate points and choose the closest depth
    # inds = sub2ind(depthOut.shape, good_uvz[:, 1].astype(np.int), good_uvz[:, 0].astype(np.int))
    # dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    #
    # # this for-loop is the most time consuming
    # import time
    # t2 = time.time()
    # for dd in dupe_inds:
    #     t1 = time.time()
    #     print("Forloop time: {}".format((t1 - t2)*1000))
    #     pts = np.where(inds == dd)[0]
    #     v_loc = int(good_uvz[pts[0], 1])
    #     u_loc = int(good_uvz[pts[0], 0])
    #     depthOut[v_loc, u_loc] = good_uvz[pts, 2].min()
    #     t2 = time.time()
    #     print("Process time: {}".format((t2 - t1)*1000))

    # Fix weird values...
    depthOut[depthOut > maxDepth] = maxDepth
    depthOut[depthOut < 0] = 0
    # depthOut[isnan(depthOut)] = 0

    return rgbUndistort, depthOriginal, depthOut


if __name__ == "__main__":
    import imageio

    # image = imageio.imread('F:/nuy_depth_v2/raw/bedroom_0001/r-1294886360.208451-2996770081.ppm')
    # depth = imageio.imread('F:/nuy_depth_v2/raw/bedroom_0001/d-1294886360.186556-2995096231.pgm')
    image = imageio.imread('/mnt/dDATASETS/nuy_depth_v2/raw/bedroom_0001/r-1294886360.208451-2996770081.ppm')
    depth_raw = imageio.imread('/mnt/dDATASETS/nuy_depth_v2/raw/bedroom_0001/d-1294886360.186556-2995096231.pgm')

    rgb_undistort, depth, depth_prj = project_depth_map(image, depth_raw)

    dep = depth_prj * 255 / np.max(depth_prj)
    dep = dep.astype(np.uint8)
    imageio.imsave("rgb.png", rgb_undistort)
    imageio.imsave("dep.png", dep)

    # plt.subplot(221)
    # plt.imshow(image)
    # plt.subplot(222)
    # plt.imshow(depth)
    # plt.subplot(223)
    # plt.imshow(rgb_undistort)
    # plt.subplot(224)
    # plt.imshow(depth_prj)
    # plt.show()
