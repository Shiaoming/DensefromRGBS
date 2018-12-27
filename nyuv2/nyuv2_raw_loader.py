from __future__ import division
import numpy as np
from path import Path
import imageio
import time
import sys

sys.path.append('../')
from nyuv2.nyuv2_utility import project_depth_map
import matplotlib.pyplot as plt


class NYUV2RawLoader(object):
    def __init__(self,
                 dataset_dir,
                 get_acc=False):
        self.dataset_dir = Path(dataset_dir)
        self.get_acc = get_acc

        self.collect_files()

    def collect_files(self):
        self.scenes = []
        scene_set = self.dataset_dir.dirs()
        for scene_dir in scene_set:
            s = self.collect_scenes(scene_dir)
            if len(s) > 0:
                self.scenes.append(s)

        self.total_samples = 0
        # count the samples
        for s in self.scenes:
            s['acc_num'] = self.total_samples
            self.total_samples += len(s['data'])

    def collect_scenes(self, scene_dir):
        scenes_index_file = scene_dir / 'INDEX.txt'
        indices = open(scenes_index_file)

        i = 0
        scene_data = {'dir': scene_dir, 'num': 0, 'data': []}
        frame = {}
        for line in indices:
            if line.startswith('d-'):
                frame['depth'] = line.strip()
            elif line.startswith('r-'):
                frame['rgb'] = line.strip()

                depth_timestamp = float(frame['depth'].split('-')[1])
                rgb_timestamp = float(frame['rgb'].split('-')[1])

                # 50ms
                if (abs(depth_timestamp - rgb_timestamp) < 0.05):
                    i += 1
                    scene_data['data'].append(frame.copy())
                    frame.clear()

        return scene_data

    def get_one_example(self):
        idx = np.random.randint(0, self.total_samples)

        img, depth_prj = None, None
        for s in self.scenes:
            if idx > s['acc_num'] and idx <= s['acc_num'] + len(s['data']):
                idx = idx - s['acc_num'] - 1
                print("idx: {}".format(idx))
                print(s['dir'] / s['data'][idx]['rgb'])
                print(s['dir'] / s['data'][idx]['depth'])
                img_raw = imageio.imread(s['dir'] / s['data'][idx]['rgb'])
                depth_raw = imageio.imread(s['dir'] / s['data'][idx]['depth'])
                img, depth, depth_prj = project_depth_map(img_raw, depth_raw)
                break

        return img, depth_prj

if __name__ == "__main__":
    data_loader = NYUV2RawLoader('F:/nuy_depth_v2/raw')
    # data_loader = NYUV2RawLoader('/mnt/dDATASETS/nuy_depth_v2/raw')

    print("Total samples: {}".format(data_loader.total_samples))

    Path.makedirs_p(Path('../nyuv2_test_imgs'))

    for i in range(10):
        t1 = time.time()
        img, depth = data_loader.get_one_example()
        t2 = time.time()
        print("Load time: {}s".format(t2 - t1))

        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(depth)
        plt.show()

        np.save('../nyuv2_test_imgs/image{}.npy'.format(i), img)
        np.save('../nyuv2_test_imgs/depth{}.npy'.format(i), depth)
