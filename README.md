### Main work

##### Dataset loader
- KITTI

The undistort kitti image size is approximately `1242x375`,
so one can not obtain crop image with size `480x640` from the original image.
There will be a resize in image and the lidar points can not fill upper part of the image.
- NYUv2

In NYUv2 there are only 1449 clean RGBD image in file `nyu_depth_v2_labeled.mat`,
but the raw depth in raw data projected to rgb image plane have zeros areas around the border.
After crop using official tool box, the image size should be `561x427`.
By the way, the paper says "missing
depth values are filled using *Colorization using optimization*", which not implemented here.

##### NN fill
Tested on KITTI image with size `240x320`,
It's time consouming with size `480x640` in kitti since the upper half part of depth is all zero.

The depth image nn fill result with downsample rate `48x48` on kitti `480x640` depth image:

![kitti480](https://raw.githubusercontent.com/Shiaoming/DensefromRGBS/master/asset/kitti480.png) 

Works fine in NYUv2.

with downsample rate `24x24`:

![nyu24](https://raw.githubusercontent.com/Shiaoming/DensefromRGBS/master/asset/nyu24.png) 

with downsample rate `48x48`:

![nyu48](https://raw.githubusercontent.com/Shiaoming/DensefromRGBS/master/asset/nyu48.png) 

##### Model
Some block details are not clear enough in the paper, so this implementation can be just a reference.
There seems no perdition of hierarchical residual depth output.
Moreover, the description of loss function is just one line explanation
"*We use Adam [15] as our optimizer, and standard pixel-wise L2 loss to train*".

##### Train
Add train scripts for NYUv2 dataset.

### Setup

```bash
git clone https://github.com/Shiaoming/DensefromRGBS.git
cd DensefromRGBS
./setup.sh
```
