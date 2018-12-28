### Brief
Implementation of paper "Estimating Depth from RGB and Sparse Sensing" with pytorch.

### Main work
##### Dataset loader
- KITTI: The undistort kitti image size is approximately `1242x375`,
so one can not obtain crop image with size `480x640` from the original image.
There will be resize in image and the lidar points can not fill this image.
- NYUv2: In NYUv2 there only 1449 full filled depth image in file `nyu_depth_v2_labeled.mat`,
the raw depth in raw data have zeros areas. By the way, the paper says "missing
depth values are filled using *Colorization using optimization*", not implement yet.
##### NN fill
Tested on KITTI image with size 240x320,
It's time consouming with size 480x640 in kitti since the upper half part of depth is all zero.
Works fine in NYUv2.
##### Model
Some block details are not clear enough about the module, so this implementation is just a reference.
 There seems no perdition of hierarchical residual depth output. Moreover, the description of loss function is just one line explain "*We
use Adam [15] as our optimizer, and standard pixel-wise L2 loss to train*".
##### Train
Not implement yet.

### Setup

```bash
git clone https://github.com/Shiaoming/DensefromRGBS.git
cd DensefromRGBS
python setup.py build_ext --inplace
```