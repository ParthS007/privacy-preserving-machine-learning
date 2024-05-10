from collections import OrderedDict

import torch
import torch.nn as nn
from ffc import FFC_BN_ACT, ConcatTupleLayer


def get_model(model_name, in_channels=1, num_classes=9, ratio=0.5):
    if model_name == "unet":
        model = UNet(in_channels, num_classes)
    elif model_name == "y_net_gen":
        model = YNet_general(in_channels, num_classes, ffc=False)
    elif model_name == "y_net_gen_ffc":
        model = YNet_general(in_channels, num_classes, ffc=True, ratio_in=ratio)
    elif model_name == "UNetOrg":
        model = UNetOrg(in_channels, num_classes)
    elif model_name == "ReLayNet":
        model = ReLayNet(in_channels, num_classes)
    elif model_name == "LFUNet":
        model = LFUNet(in_channels, num_classes)
    elif model_name == "FCN8s":
        model = FCN8s(in_channels, num_classes)
    elif model_name == "NestedUNet":
        model = NestedUNet(in_channels, num_classes)

    else:
        print("Model name not found")
        assert False

    return model


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, num_groups=8):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        # self.bn1 = nn.BatchNorm2d(middle_channels)
        self.bn1 = nn.GroupNorm(num_groups, middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(
        self, input_channels=3, num_classes=10, deep_supervision=False, **kwargs
    ):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(
            nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0]
        )
        self.conv1_2 = VGGBlock(
            nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1]
        )
        self.conv2_2 = VGGBlock(
            nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2]
        )

        self.conv0_3 = VGGBlock(
            nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0]
        )
        self.conv1_3 = VGGBlock(
            nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1]
        )

        self.conv0_4 = VGGBlock(
            nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0]
        )

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    # (name + "norm1", nn.BatchNorm2d(num_features=features)),  # non private
                    (name + "norm1", nn.GroupNorm(32, features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    # (name + "norm2", nn.BatchNorm2d(num_features=features)), # non private
                    (name + "norm2", nn.GroupNorm(32, features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class BasicUNet(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, init_features=64, conv_ker=(3, 3)
    ):
        super(BasicUNet, self).__init__()

        conv_pad = (int((conv_ker[0] - 1) / 2), int((conv_ker[1] - 1) / 2))
        features = init_features
        self.encoder1 = BasicUNet._block(
            in_channels, features, name="enc1", conv_kernel=conv_ker, conv_pad=conv_pad
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.encoder2 = BasicUNet._block(
            features, features * 2, name="enc2", conv_kernel=conv_ker, conv_pad=conv_pad
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.encoder3 = BasicUNet._block(
            features * 2,
            features * 4,
            name="enc3",
            conv_kernel=conv_ker,
            conv_pad=conv_pad,
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.encoder4 = BasicUNet._block(
            features * 4,
            features * 8,
            name="enc4",
            conv_kernel=conv_ker,
            conv_pad=conv_pad,
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.bottleneck = BasicUNet._block(
            features * 8,
            features * 16,
            name="bottleneck",
            conv_kernel=conv_ker,
            conv_pad=conv_pad,
        )

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=(2, 2), stride=2
        )
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder4 = BasicUNet._block(
            (features * 8) * 2,
            features * 8,
            name="dec4",
            conv_kernel=conv_ker,
            conv_pad=conv_pad,
        )
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=(2, 2), stride=2
        )
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder3 = BasicUNet._block(
            (features * 4) * 2,
            features * 4,
            name="dec3",
            conv_kernel=conv_ker,
            conv_pad=conv_pad,
        )
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=(2, 2), stride=2
        )
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder2 = BasicUNet._block(
            (features * 2) * 2,
            features * 2,
            name="dec2",
            conv_kernel=conv_ker,
            conv_pad=conv_pad,
        )
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=(2, 2), stride=2
        )
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder1 = BasicUNet._block(
            features * 2, features, name="dec1", conv_kernel=conv_ker, conv_pad=conv_pad
        )

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=(1, 1)
        )
        self.softmax = nn.Softmax2d()

    def forward_encoder(self, x):
        enc1 = self.encoder1(x)
        pool1, indices1 = self.pool1(enc1)
        enc2 = self.encoder2(pool1)
        pool2, indices2 = self.pool2(enc2)
        enc3 = self.encoder3(pool2)
        pool3, indices3 = self.pool3(enc3)
        enc4 = self.encoder4(pool3)
        pool4, indices4 = self.pool4(enc4)
        bottleneck = self.bottleneck(pool4)
        return (
            enc1,
            pool1,
            indices1,
            enc2,
            pool2,
            indices2,
            enc3,
            pool3,
            indices3,
            enc4,
            pool4,
            indices4,
            bottleneck,
        )

    def forward_decoder(self, enc1, enc2, enc3, enc4, bottleneck):
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return dec1

    def forward(self, x):
        (
            enc1,
            pool1,
            indices1,
            enc2,
            pool2,
            indices2,
            enc3,
            pool3,
            indices3,
            enc4,
            pool4,
            indices4,
            bottleneck,
        ) = BasicUNet.forward_encoder(self, x)
        dec1 = BasicUNet.forward_decoder(self, enc1, enc2, enc3, enc4, bottleneck)
        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name, conv_kernel, conv_pad, repetitions=2):
        block_sequence = nn.Sequential()
        for count in range(repetitions):
            block_sequence.add_module(
                name + "conv" + str(count + 1),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=features,
                    kernel_size=conv_kernel,
                    padding=conv_pad,
                    bias=False,
                ),
            )

            block_sequence.add_module(
                name + "norm" + str(count + 1), nn.GroupNorm(32, features)
            )
            block_sequence.add_module(
                name + "relu" + str(count + 1), nn.ReLU(inplace=True)
            )
            in_channels = features
        return block_sequence


# Original UNet - just use the baseline model
# O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation.” 2015.
class UNetOrg(BasicUNet):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__(
            in_channels=in_channels,
            out_channels=num_classes,
            init_features=32,
            conv_ker=(3, 3),
        )


# ReLayNet - first UNet adapted for retina layers segmentation
# A. Roy, et al., “ReLayNet: retinal layer and fluid segmentation of macular optical coherence tomography using
# fully convolutional networks,” Biomed. Opt. Express, vol. 8, no. 8, pp. 3627–3642, Aug. 2017.
class ReLayNet(BasicUNet):
    def __init__(self, in_channels=3, num_classes=10, features=64, kernel=(7, 3)):
        super().__init__(
            in_channels=in_channels,
            out_channels=num_classes,
            init_features=features,
            conv_ker=kernel,
        )

        conv_pad = (int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))

        self.encoder1 = ReLayNet._block(
            in_channels,
            features,
            name="enc1",
            conv_kernel=kernel,
            conv_pad=conv_pad,
            repetitions=1,
        )
        self.encoder2 = ReLayNet._block(
            features,
            features,
            name="enc2",
            conv_kernel=kernel,
            conv_pad=conv_pad,
            repetitions=1,
        )
        self.encoder3 = ReLayNet._block(
            features,
            features,
            name="enc3",
            conv_kernel=kernel,
            conv_pad=conv_pad,
            repetitions=1,
        )

        self.bottleneck = ReLayNet._block(
            features,
            features,
            name="bottleneck",
            conv_kernel=kernel,
            conv_pad=conv_pad,
            repetitions=1,
        )

        self.decoder3 = ReLayNet._block(
            features * 2,
            features,
            name="dec3",
            conv_kernel=kernel,
            conv_pad=conv_pad,
            repetitions=1,
        )
        self.decoder2 = ReLayNet._block(
            features * 2,
            features,
            name="dec2",
            conv_kernel=kernel,
            conv_pad=conv_pad,
            repetitions=1,
        )
        self.decoder1 = ReLayNet._block(
            features * 2,
            features,
            name="dec1",
            conv_kernel=kernel,
            conv_pad=conv_pad,
            repetitions=1,
        )
        self.softmax = nn.Softmax2d()

    def forward_encoder(self, x):
        enc1 = self.encoder1(x)
        pool1, indices1 = self.pool1(enc1)
        enc2 = self.encoder2(pool1)
        pool2, indices2 = self.pool2(enc2)
        enc3 = self.encoder3(pool2)
        pool3, indices3 = self.pool3(enc3)
        bottleneck = self.bottleneck(pool3)
        return (
            enc1,
            pool1,
            indices1,
            enc2,
            pool2,
            indices2,
            enc3,
            pool3,
            indices3,
            bottleneck,
        )

    def forward_decoder(
        self,
        enc1,
        pool1,
        indices1,
        enc2,
        pool2,
        indices2,
        enc3,
        pool3,
        indices3,
        bottleneck,
    ):
        # enc1, pool1, indices1, enc2, pool2, indices2, enc3, pool3, indices3, bottleneck = ReLayNet.forward_encoder(self, x)

        dec3 = self.unpool3(bottleneck, indices3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.unpool2(dec3, indices2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.unpool1(dec2, indices1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        # dec1 = self.decoder1(dec1)

        # return self.softmax(self.conv(dec1))
        return self.decoder1(dec1)

    def forward(self, x):
        (
            enc1,
            pool1,
            indices1,
            enc2,
            pool2,
            indices2,
            enc3,
            pool3,
            indices3,
            bottleneck,
        ) = ReLayNet.forward_encoder(self, x)
        dec1 = ReLayNet.forward_decoder(
            self,
            enc1,
            pool1,
            indices1,
            enc2,
            pool2,
            indices2,
            enc3,
            pool3,
            indices3,
            bottleneck,
        )
        return self.softmax(self.conv(dec1))


# LF-UNet - dual network combining UNet and FCN
# D. Ma, et al. “Cascade Dual-branch Deep Neural Networks for Retinal Layer and fluid Segmentation of Optical
# Coherence Tomography Incorporating Relative Positional Map,” in Proceedings of the Third
# Conference on Medical Imaging with Deep Learning, vol. 121, pp. 493–502, 2020.
class LFUNet(BasicUNet):
    def __init__(self, in_channels=3, num_classes=10, features=64, kernel=(7, 3)):
        super().__init__(
            in_channels=in_channels,
            out_channels=num_classes,
            init_features=features,
            conv_ker=kernel,
        )

        conv_pad = (int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))

        self.upconv4a = nn.ConvTranspose2d(
            features * 16, features * 4, kernel_size=(2, 2), stride=2
        )
        self.upconv4b = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=(2, 2), stride=2
        )
        self.upconv4c = nn.ConvTranspose2d(
            features * 2, features, kernel_size=(2, 2), stride=2
        )
        self.upconv4d = nn.ConvTranspose2d(
            features, features, kernel_size=(2, 2), stride=2
        )

        dilation4 = (kernel[0] + 1, kernel[1] + 1)
        dilation6 = (kernel[0] + 3, kernel[1] + 3)
        dilation8 = (kernel[0] + 5, kernel[1] + 5)
        paddil4 = (conv_pad[0] * dilation4[0], conv_pad[1] * dilation4[1])
        paddil6 = (conv_pad[0] * dilation6[0], conv_pad[1] * dilation6[1])
        paddil8 = (conv_pad[0] * dilation8[0], conv_pad[1] * dilation8[1])
        self.convdil4 = nn.Conv2d(
            features * 2,
            int(features / 2),
            kernel_size=kernel,
            padding=paddil4,
            bias=False,
            dilation=dilation4,
        )
        self.convdil6 = nn.Conv2d(
            features * 2,
            int(features / 2),
            kernel_size=kernel,
            padding=paddil6,
            bias=False,
            dilation=dilation6,
        )
        self.convdil8 = nn.Conv2d(
            features * 2,
            int(features / 2),
            kernel_size=kernel,
            padding=paddil8,
            bias=False,
            dilation=dilation8,
        )

        self.dropout = nn.Dropout2d()
        self.conv = nn.Conv2d(
            in_channels=int(features / 2 * 3),
            out_channels=num_classes,
            kernel_size=(1, 1),
        )

    def forward(self, x):
        (
            enc1,
            pool1,
            indices1,
            enc2,
            pool2,
            indices2,
            enc3,
            pool3,
            indices3,
            enc4,
            pool4,
            indices4,
            bottleneck,
        ) = LFUNet.forward_encoder(self, x)
        dec1 = LFUNet.forward_decoder(self, enc1, enc2, enc3, enc4, bottleneck)

        fcn4 = self.upconv4a(bottleneck) + pool3
        fcn3 = self.upconv4b(fcn4) + pool2
        fcn2 = self.upconv4c(fcn3) + pool1
        fcn1 = self.upconv4d(fcn2)

        fcn = torch.cat((dec1, fcn1), dim=1)
        dil8 = self.convdil8(fcn)
        dil6 = self.convdil6(fcn)
        dil4 = self.convdil4(fcn)

        last = torch.cat((dil8, dil6, dil4), dim=1)
        drop = self.dropout(last)

        # return self.conv(drop)
        return self.softmax(self.conv(drop))


class FCN8s(nn.Module):

    def __init__(self, in_channels=3, num_classes=4, features=64, kernel=(3, 3)):
        super().__init__()
        conv_pad = (int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))

        # conv1
        self.conv1_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=features,
            kernel_size=kernel,
            padding=(100, 100),
        )
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(
            in_channels=features,
            out_channels=features,
            kernel_size=kernel,
            padding=conv_pad,
        )
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # conv2
        self.conv2_1 = nn.Conv2d(
            in_channels=features,
            out_channels=(features * 2),
            kernel_size=kernel,
            padding=conv_pad,
        )
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(
            in_channels=(features * 2),
            out_channels=(features * 2),
            kernel_size=kernel,
            padding=conv_pad,
        )
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # conv3
        self.conv3_1 = nn.Conv2d(
            in_channels=(features * 2),
            out_channels=(features * 4),
            kernel_size=kernel,
            padding=conv_pad,
        )
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(
            in_channels=(features * 4),
            out_channels=(features * 4),
            kernel_size=kernel,
            padding=conv_pad,
        )
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(
            in_channels=(features * 4),
            out_channels=(features * 4),
            kernel_size=kernel,
            padding=conv_pad,
        )
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(
            in_channels=(features * 4),
            out_channels=(features * 8),
            kernel_size=kernel,
            padding=conv_pad,
        )
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(
            in_channels=(features * 8),
            out_channels=(features * 8),
            kernel_size=kernel,
            padding=conv_pad,
        )
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(
            in_channels=(features * 8),
            out_channels=(features * 8),
            kernel_size=kernel,
            padding=conv_pad,
        )
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(
            in_channels=(features * 8),
            out_channels=(features * 8),
            kernel_size=kernel,
            padding=conv_pad,
        )
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(
            in_channels=(features * 8),
            out_channels=(features * 8),
            kernel_size=kernel,
            padding=conv_pad,
        )
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(
            in_channels=(features * 8),
            out_channels=(features * 8),
            kernel_size=kernel,
            padding=conv_pad,
        )
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(
            in_channels=(features * 8),
            out_channels=((features * 8) * 8),
            kernel_size=(7, 7),
        )
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(
            in_channels=((features * 8) * 8),
            out_channels=((features * 8) * 8),
            kernel_size=(1, 1),
        )
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_conv7 = nn.Conv2d(
            in_channels=((features * 8) * 8),
            out_channels=num_classes,
            kernel_size=(1, 1),
        )
        self.score_pool4 = nn.Conv2d(
            in_channels=(features * 8), out_channels=num_classes, kernel_size=(1, 1)
        )
        self.score_pool3 = nn.Conv2d(
            in_channels=(features * 4), out_channels=num_classes, kernel_size=(1, 1)
        )

        self.upscore_conv7 = nn.ConvTranspose2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=4,
            stride=2,
            bias=False,
        )
        self.upscore_pool4 = nn.ConvTranspose2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=4,
            stride=2,
            bias=False,
        )
        self.upscore_final = nn.ConvTranspose2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=16,
            stride=8,
            bias=False,
        )

    def forward(self, x):
        conv1 = self.relu1_2(self.conv1_2(self.relu1_1(self.conv1_1(x))))
        pool1 = self.pool1(conv1)

        conv2 = self.relu2_2(self.conv2_2(self.relu2_1(self.conv2_1(pool1))))
        pool2 = self.pool2(conv2)

        conv3 = self.relu3_3(
            self.conv3_3(self.relu3_2(self.conv3_2(self.relu3_1(self.conv3_1(pool2)))))
        )
        pool3 = self.pool3(conv3)

        conv4 = self.relu4_3(
            self.conv4_3(self.relu4_2(self.conv4_2(self.relu4_1(self.conv4_1(pool3)))))
        )
        pool4 = self.pool4(conv4)

        conv5 = self.relu5_3(
            self.conv5_3(self.relu5_2(self.conv5_2(self.relu5_1(self.conv5_1(pool4)))))
        )
        pool5 = self.pool5(conv5)

        conv6 = self.relu6(self.fc6(pool5))
        drop6 = self.drop6(conv6)

        conv7 = self.relu7(self.fc7(drop6))
        drop7 = self.drop7(conv7)

        upscore2 = self.upscore_conv7(self.score_conv7(drop7))  # 1/16
        score_pool4 = self.score_pool4(pool4)
        score_pool4c = score_pool4[
            :, :, 5 : (5 + upscore2.size()[2]), 5 : (5 + upscore2.size()[3])
        ]  # 1/8
        upscore_pool4 = self.upscore_pool4(upscore2 + score_pool4c)
        score_pool3 = self.score_pool3(pool3)
        score_pool3c = score_pool3[
            :, :, 9 : (9 + upscore_pool4.size()[2]), 9 : (9 + upscore_pool4.size()[3])
        ]  # 1/8

        final = self.upscore_final(upscore_pool4 + score_pool3c)
        output = final[
            :, :, 31 : (31 + x.size()[2]), 31 : (31 + x.size()[3])
        ].contiguous()

        return output


class YNet_general(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        init_features=32,
        ratio_in=0.5,
        ffc=True,
        skip_ffc=False,
        cat_merge=True,
    ):
        super(YNet_general, self).__init__()

        self.ffc = ffc
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge

        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_general._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_general._block(
            features, features * 2, name="enc2"
        )  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_general._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_general._block(
            features * 4, features * 4, name="enc4"
        )  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(
                in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in
            )
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(
                features,
                features * 2,
                kernel_size=1,
                ratio_gin=ratio_in,
                ratio_gout=ratio_in,
            )  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(
                features * 2,
                features * 4,
                kernel_size=1,
                ratio_gin=ratio_in,
                ratio_gout=ratio_in,
            )
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(
                features * 4,
                features * 4,
                kernel_size=1,
                ratio_gin=ratio_in,
                ratio_gout=ratio_in,
            )  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_general._block(in_channels, features, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_general._block(
                features, features * 2, name="enc2_2"
            )  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_general._block(
                features * 2, features * 4, name="enc3_2"
            )  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_general._block(
                features * 4, features * 4, name="enc4_2"
            )  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = YNet_general._block(
            features * 8, features * 16, name="bottleneck"
        )  # 8, 16

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general._block(
                (features * 8) * 2, features * 8, name="dec4"
            )  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general._block(
                (features * 6) * 2, features * 4, name="dec3"
            )
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general._block(
                (features * 3) * 2, features * 2, name="dec2"
            )
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general._block(
                features * 3, features, name="dec1"
            )  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general._block(
                (features * 8) * 2, features * 8, name="dec4"
            )  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general._block(
                (features * 6) * 2, features * 4, name="dec3"
            )
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general._block(
                (features * 3) * 2, features * 2, name="dec2"
            )
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general._block(
                features * 3, features, name="dec1"
            )  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general._block(
                (features * 6) * 2, features * 8, name="dec4"
            )  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general._block(
                (features * 4) * 2, features * 4, name="dec3"
            )
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general._block(
                (features * 2) * 2, features * 2, name="dec2"
            )
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general._block(
                features * 2, features, name="dec1"
            )  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(
            (
                batch,
                -1,
            )
            + ffted.size()[3:]
        )
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))
        enc4_2 = self.pool4(enc4)

        if self.ffc:
            enc1_f = self.encoder1_f(x)
            enc1_l, enc1_g = enc1_f
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))

        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.cat_merge:
            a = torch.zeros_like(enc4_2)
            b = torch.zeros_like(enc4_f2)

            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1)
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)

            bottleneck = torch.cat((enc4_2, enc4_f2), 1)
            bottleneck = bottleneck.view_as(torch.cat((a, b), 1))

        else:
            bottleneck = torch.cat((enc4_2, enc4_f2), 1)

        bottleneck = self.bottleneck(bottleneck)

        dec4 = self.upconv4(bottleneck)

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.GroupNorm(32, features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.GroupNorm(32, features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
