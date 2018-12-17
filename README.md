### Brief
Implementation of paper "Estimating Depth from RGB and Sparse Sensing".

### TODO

- [x] NN fill : tested on KITTI image with size 240x320, 
it's time consouming with size 480x640 since the upper half part of depth is all zero.
- [ ] model
- [ ] train
- [ ] eval

### setup

```bash
git clone https://github.com/Shiaoming/DensefromRGBS.git
cd DensefromRGBS
python setup.py build_ext --inplace
```