### Brief
Implementation of paper "Estimating Depth from RGB and Sparse Sensing".

### TODO

- [x] NN fill : Tested on KITTI image with size 240x320,
it's time consouming with size 480x640 since the upper half part of depth is all zero.
- [x] model : Some block details are not clear enough about the module, so this implementation is just a reference.
 There seems no perdition of hierarchical residual depth output. Moreover, the description of loss function is just one line explain "*We
use Adam [15] as our optimizer, and standard pixel-wise L2 loss to train*".
- [ ] train

### setup

```bash
git clone https://github.com/Shiaoming/DensefromRGBS.git
cd DensefromRGBS
python setup.py build_ext --inplace
```