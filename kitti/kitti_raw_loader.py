'''
Modified from https://github.com/tinghuiz/SfMLearner/blob/master/data/kitti/kitti_raw_loader.py
'''
from __future__ import division
import numpy as np
from path import Path
import scipy.misc
from collections import Counter
import sys
import copy

import matplotlib.pyplot as plt

sys.path.append('../')
from kitti.viz_utility import *


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def pose_from_oxts_packet(metadata, scale):
    lat, lon, alt, roll, pitch, yaw = metadata
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    Taken from https://github.com/utiasSTARS/pykitti
    """

    er = 6378137.  # earth radius (approx.) in meters
    # Use a Mercator projection to get the translation vector
    ty = lat * np.pi * er / 180.

    tx = scale * lon * np.pi * er / 180.
    # ty = scale * er * \
    #     np.log(np.tan((90. + lat) * np.pi / 360.))
    tz = alt
    t = np.array([tx, ty, tz]).reshape(-1, 1)

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(roll)
    Ry = roty(pitch)
    Rz = rotz(yaw)
    R = Rz.dot(Ry.dot(Rx))
    return transform_from_rot_trans(R, t)


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


class KittiRawLoader(object):
    def __init__(self,
                 dataset_dir,
                 read_right_img=False,
                 img_height=480,
                 img_width=640,
                 test_percentage=0.1,
                 get_depth=True,
                 get_pose=False):

        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        if read_right_img:
            self.cam_ids = ['02', '03']
        else:
            self.cam_ids = ['02']
        self.date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        self.test_percentage = test_percentage
        self.get_depth = get_depth
        self.get_pose = get_pose
        self.collect_files()

    def collect_files(self):
        self.scenes = []
        for date in self.date_list:
            drive_set = (self.dataset_dir / date).dirs()
            for dr in drive_set:
                s = self.collect_scenes(dr)
                if len(s) > 0:
                    self.scenes.append(s)

        self.total_samples = 0
        # count the samples
        for s in self.scenes:
            s[0]['acc_num'] = self.total_samples
            self.total_samples += s[0]['num']

    def collect_scenes(self, drive):
        scenes = []
        for c in self.cam_ids:
            # scene_data = {'cid': c, 'dir': drive, 'speed': [], 'frame_id': [], 'pose':[], 'rel_path': drive.name + '_' + c}
            scene_data = {'cid': c, 'dir': drive, 'num': 0, 'w': 0, 'h': 0}

            cam2cam = read_calib_file(drive.parent / 'calib_cam_to_cam.txt')
            velo2cam = read_calib_file(drive.parent / 'calib_velo_to_cam.txt')

            cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))
            velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])

            scene_data['w'] = cam2cam['S_rect_' + scene_data['cid']][0]
            scene_data['h'] = cam2cam['S_rect_' + scene_data['cid']][1]

            oxts = (drive / 'oxts' / 'data').files('*.txt')
            scene_data['num'] = len(oxts)

            if self.get_pose:
                oxts = sorted(oxts)
                scale = None
                origin = None
                imu2velo = read_calib_file(drive.parent / 'calib_imu_to_velo.txt')

                imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])

                imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat

                for n, f in enumerate(oxts):
                    metadata = np.genfromtxt(f)
                    # speed = metadata[8:11]
                    # scene_data['speed'].append(speed)
                    # scene_data['frame_id'].append('{:010d}'.format(n))
                    lat = metadata[0]

                    if scale is None:
                        scale = np.cos(lat * np.pi / 180.)

                    pose_matrix = pose_from_oxts_packet(metadata[:6], scale)
                    if origin is None:
                        origin = pose_matrix

                    odo_pose = imu2cam @ np.linalg.inv(origin) @ pose_matrix @ np.linalg.inv(imu2cam)
                    scene_data['pose'].append(odo_pose[:3])

            # check empty scene
            sample = (drive / 'image_{}'.format(scene_data['cid']) / 'data').files('*.png')
            if len(sample) == 0:
                return []

            scene_data['velo2cam'] = velo2cam
            scene_data['cam2cam'] = cam2cam
            scene_data['P_rect'] = self.get_P_rect(scene_data)
            # scene_data['intrinsics'] = scene_data['P_rect'][:, :3]

            scenes.append(scene_data)
        return scenes

    def get_P_rect(self, scene_data):
        calib_file = scene_data['dir'].parent / 'calib_cam_to_cam.txt'

        filedata = self.read_raw_calib_file(calib_file)
        P_rect = np.reshape(filedata['P_rect_' + scene_data['cid']], (3, 4))
        return P_rect

    def read_raw_calib_file(self, filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def load_image(self, scene, idx):
        '''
        If the target height larger than image height, stretch the image height to target height and crop the width to
        keep target ratio.
        Otherwise just crop from the original image(crop from bottom for cover as much Velodyne points as possible).
        :param scene: current scene
        :param idx: image index
        :return: image and P_rect
        '''
        img_file = scene['dir'] / 'image_{}'.format(scene['cid']) / 'data' / '{:010d}.png'.format(idx)
        if not img_file.isfile():
            return None
        img = scipy.misc.imread(img_file)
        P_rect = np.copy(scene['P_rect'])

        if self.img_height > img.shape[0]:
            # resize
            zoom = self.img_height / img.shape[0]
            new_width = int(img.shape[1] * zoom)
            img = scipy.misc.imresize(img, (self.img_height, new_width))
            P_rect[0] *= zoom
            P_rect[1] *= zoom

            # random crop
            offset_w = np.random.randint(0, new_width - self.img_width + 1)
            img = img[:, offset_w:offset_w + self.img_width]
            P_rect[0, 2] -= offset_w
        else:
            # random crop
            offset_h = img.shape[0] - self.img_height
            offset_w = np.random.randint(0, img.shape[1] - self.img_width + 1)
            img = img[offset_h:offset_h + self.img_height, offset_w:offset_w + self.img_width]
            P_rect[0, 2] -= offset_w
            P_rect[1, 2] -= offset_h

        return img, P_rect

    def generate_depth_map(self, scene, idx, crop_P_rect):
        # compute projection matrix velodyne->image plane

        def sub2ind(matrixSize, rowSub, colSub):
            m, n = matrixSize
            return rowSub * (n - 1) + colSub - 1

        R_cam2rect = np.eye(4)

        cam2cam = scene['cam2cam']
        velo2cam = scene['velo2cam']
        velo2cam = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])  # 4x4

        P_rect = np.copy(crop_P_rect)  # 3x4

        R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)  # 4x4

        # P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
        P_velo2im = P_rect @ R_cam2rect @ velo2cam  # 3x4

        velo_file_name = scene['dir'] / 'velodyne_points' / 'data' / '{:010d}.bin'.format(idx)

        # load velodyne points and remove all behind image plane (approximation)
        # each row of the velodyne data is forward, left, up, reflectance
        velo = load_velo_scan(velo_file_name)  # nx4
        # original_velo = copy.deepcopy(velo)
        velo[:, 3] = 1
        velo = velo[velo[:, 0] >= 0, :]  # x>0 means all the points in front

        # project the points to the camera
        # velo_pts_im = np.dot(P_velo2im, velo.T).T
        velo_pts_im = (P_velo2im @ velo.T).T  # nx3
        velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, -1:]  # (x/z,y/z,z)

        # check if in bounds
        # use minus 1 to get the exact same value as KITTI matlab code
        velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
        velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1

        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = val_inds & (velo_pts_im[:, 0] < self.img_width)
        val_inds = val_inds & (velo_pts_im[:, 1] < self.img_height)
        velo_pts_im = velo_pts_im[val_inds, :]

        # crop_velo = velo[val_inds, :]

        # project to image
        depth = np.zeros((self.img_height, self.img_width)).astype(np.float32)
        depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

        # find the duplicate points and choose the closest depth
        inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
        depth[depth < 0] = 0
        return depth, velo_pts_im  # original_velo, crop_velo

    def get_one_example(self):
        idx = np.random.randint(0, self.total_samples)

        # print("idx: {}".format(idx))
        img, depth, velo_pts_im = None, None, None
        for s in self.scenes:
            if idx > s[0]['acc_num'] and idx <= s[0]['acc_num'] + s[0]['num']:
                idx = idx - s[0]['acc_num'] - 1
                # print("Scene {}, idx {}".format(s[0]['dir'], idx))
                img, P = self.load_image(s[0], idx)
                depth, velo_pts_im = self.generate_depth_map(s[0], idx, P)
                break

        return img, depth, velo_pts_im


if __name__ == "__main__":

    data_loader = KittiRawLoader('/home/zxm/dataset/kitti_raw', img_height=240, img_width=320)

    print("Total samples: {}".format(data_loader.total_samples))

    Path.makedirs_p(Path('../kitti_test_imgs'))

    for i in range(10):
        img, depth, ptsim = data_loader.get_one_example()

        # plt.subplot(211)
        # plt.imshow(img)
        # plt.subplot(212)
        # plt.imshow(depth)
        # plt.show()

        # save for latter test
        np.save('../kitti_test_imgs/image{}.npy'.format(i), img)
        np.save('../kitti_test_imgs/depth{}.npy'.format(i), depth)
        np.save('../kitti_test_imgs/ptsim{}.npy'.format(i), ptsim)

