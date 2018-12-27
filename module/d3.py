import torch
import torch.nn as nn
from module.densenet import *


class D3(nn.Module):

    def __init__(self, opts):
        super(D3, self).__init__()

        self.L = opts['L']
        self.k = opts['k']
        self.bn = opts['bn']

        self.downsample_half = nn.UpsamplingBilinear2d(scale_factor=0.5)

        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, stride=2, padding=1)

        # DenseNet Block 1
        self.d1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dense1 = DenseNet_conv(in_c=64, L=self.L, k=self.k, bn=self.bn)
        self.d1_conv2 = nn.Conv2d(self.k, 64, kernel_size=3, stride=2, padding=1)

        # DenseNet Block 2
        self.d2_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dense2 = DenseNet_conv(in_c=64, L=self.L, k=self.k, bn=self.bn)
        self.d2_conv2 = nn.Conv2d(self.k, 64, kernel_size=3, stride=2, padding=1)

        # DenseNet Block 3
        self.d3_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dense3 = DenseNet_conv(in_c=64, L=self.L, k=self.k, bn=self.bn)
        self.d3_conv2 = nn.Conv2d(self.k, 64, kernel_size=3, stride=2, padding=1)

        # DenseNet Block 4
        self.d4_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dense4 = DenseNet_conv(in_c=64, L=self.L, k=self.k, bn=self.bn)
        self.d4_conv2 = nn.Conv2d(self.k, 64, kernel_size=3, stride=2, padding=1)

        # DenseNet Block 5
        self.d5_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dense5 = DenseNet_conv(in_c=64, L=self.L, k=self.k, bn=self.bn)
        self.d5_conv2 = nn.Conv2d(self.k, 64, kernel_size=3, stride=1, padding=1)

        # DenseNet Block 6
        self.d6_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dense6 = DenseNet_conv(in_c=64, L=self.L, k=self.k, bn=self.bn)
        self.d6_conv2 = nn.Conv2d(self.k, 64, kernel_size=3, stride=1, padding=1)

        # DenseNet Connecting Block 2
        self.d2c_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dense2c = DenseNet_conv(in_c=64, L=self.L, k=self.k, bn=self.bn)
        self.d2c_conv2 = nn.Conv2d(self.k, 64, kernel_size=3, stride=1, padding=1)

        # DenseNet Connecting Block 3
        self.d3c_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dense3c = DenseNet_conv(in_c=64, L=self.L, k=self.k, bn=self.bn)
        self.d3c_conv2 = nn.Conv2d(self.k, 64, kernel_size=3, stride=1, padding=1)

        # DenseNet Connecting Block 4
        self.d4c_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dense4c = DenseNet_conv(in_c=64, L=self.L, k=self.k, bn=self.bn)
        self.d4c_conv2 = nn.Conv2d(self.k, 64, kernel_size=3, stride=1, padding=1)

        # DenseNet Block upsample 1
        self.up1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.up_dense1 = DenseNet_conv(in_c=64, L=self.L, k=self.k, bn=self.bn)
        self.up1_conv2 = nn.ConvTranspose2d(self.k, 64, kernel_size=2, stride=2, padding=0)

        # DenseNet Block upsample 2
        self.up2_conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.up_dense2 = DenseNet_conv(in_c=64, L=self.L, k=self.k, bn=self.bn)
        self.up2_conv2 = nn.ConvTranspose2d(self.k, 64, kernel_size=2, stride=2, padding=0)

        # DenseNet Block upsample 3
        self.up3_conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.up_dense3 = DenseNet_conv(in_c=64, L=self.L, k=self.k, bn=self.bn)
        self.up3_conv2 = nn.ConvTranspose2d(self.k, 64, kernel_size=2, stride=2, padding=0)

        # DenseNet Block upsample 4
        self.up4_conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.up_dense4 = DenseNet_conv(in_c=64, L=self.L, k=self.k, bn=self.bn)
        self.up4_conv2 = nn.ConvTranspose2d(self.k, 64, kernel_size=2, stride=2, padding=0)

        # Final DeConvolution
        self.up5_conv1 = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, padding=0)

    def forward(self, img, sparse_inputs):
        # Downsampling layers
        h = torch.relu(self.conv1(torch.cat((img, sparse_inputs), 1)))

        # Dense1
        x240 = self.downsample_half(sparse_inputs)
        h = torch.relu(self.d1_conv1(h))
        h = self.dense1(h, x240)
        h = torch.relu(self.d1_conv2(h))

        # Skip Dense 2
        x120 = self.downsample_half(x240)
        skip_h2 = torch.relu(self.d2c_conv1(h))
        skip_h2 = self.dense2c(skip_h2, x120)
        skip_h2 = torch.relu(self.d2c_conv2(skip_h2))

        # Dense 2
        h = torch.relu(self.d2_conv1(h))
        h = self.dense2(h, x120)
        h = torch.relu(self.d2_conv2(h))

        # Skip Dense 3
        x60 = self.downsample_half(x120)
        skip_h3 = torch.relu(self.d3c_conv1(h))
        skip_h3 = self.dense3c(skip_h3, x60)
        skip_h3 = torch.relu(self.d3c_conv2(skip_h3))

        # Dense 3
        h = torch.relu(self.d3_conv1(h))
        h = self.dense3(h, x60)
        h = torch.relu(self.d3_conv2(h))

        # Skip Dense 4
        x30 = self.downsample_half(x60)
        skip_h4 = torch.relu(self.d4c_conv1(h))
        skip_h4 = self.dense4c(skip_h4, x30)
        skip_h4 = torch.relu(self.d4c_conv2(skip_h4))

        # Dense 4
        x15 = self.downsample_half(x30)
        h = torch.relu(self.d4_conv1(h))
        h = self.dense4(h, x30)
        h = torch.relu(self.d4_conv2(h))

        # Dense 5
        h = torch.relu(self.d5_conv1(h))
        h = self.dense5(h, x15)
        h = torch.relu(self.d5_conv2(h))

        # Dense 6
        h = torch.relu(self.d6_conv1(h))
        h = self.dense6(h, x15)
        h = torch.relu(self.d6_conv2(h))

        # UpDense 1
        h = torch.relu(self.up1_conv1(h))
        h = self.up_dense1(h, x15)
        h = torch.relu(self.up1_conv2(h))

        h = torch.cat((h, skip_h4), 1)

        # UpDense 2
        h = torch.relu(self.up2_conv1(h))
        h = self.up_dense2(h, x30)
        h = torch.relu(self.up2_conv2(h))

        h = torch.cat((h, skip_h3), 1)

        # UpDense 3
        h = torch.relu(self.up3_conv1(h))
        h = self.up_dense3(h, x60)
        h = torch.relu(self.up3_conv2(h))

        h = torch.cat((h, skip_h2), 1)

        # UpDense 4
        h = torch.relu(self.up4_conv1(h))
        h = self.up_dense4(h, x120)
        h = torch.relu(self.up4_conv2(h))

        h = torch.relu(self.up5_conv1(h))

        return h


if __name__ == "__main__":
    # batch_size: 1 for test
    # input channels: rgb image is 3
    # L=5, k=12 in paper
    batch, in_c, L, k = 1, 3, 5, 12

    x = torch.randn((batch, in_c, 480, 640))
    # s is two channels, (s1,s2) in paper
    s = torch.randn((batch, 2, 480, 640))

    opt = {'L': L, 'k': k, 'bn': False}

    d3 = D3(opt)
    out = d3(x, s)

    print(d3)
    print("input x: ", x.shape)
    print("input s: ", s.shape)
    print("output: ", out.shape)
