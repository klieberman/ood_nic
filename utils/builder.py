import math

import torch
import torch.nn as nn

import utils.gmp
import utils.dense


class Builder(object):
    def __init__(self, conv_layer, deconv_layer, linear_layer):
        self.conv_layer = conv_layer
        self.deconv_layer = deconv_layer
        self.linear_layer = linear_layer

        # Amount of padding for each kernel size
        self.kernel_padding = {
            1: 0,
            3: 1,
            5: 2,
            7: 3
        }

        # Amount of (padding, output_padding) for each kernel size
        self.kernel_padding_transpose = {
            3: (1, 0),
            5: (2, 1)
        }

    def conv(self, kernel_size, in_planes, out_planes, stride=1, bias=False):
        assert kernel_size in self.kernel_padding, f"invalid kernel size {kernel_size} for conv"
        padding = self.kernel_padding[kernel_size]
        conv = self.conv_layer(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        self._init_conv(conv)

        return conv


    def deconv(self, kernel_size, in_planes, out_planes, stride=1, bias=False):  
        assert kernel_size in self.kernel_padding_transpose, f"invalid kernel size {kernel_size} for conv \
                                                            with transpose=True"
        padding, output_padding = self.kernel_padding_transpose[kernel_size]
        deconv = self.deconv_layer(
            in_planes, 
            out_planes,
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias)

        self._init_conv(deconv)

        return deconv
    

    def linear(self, in_features, out_features, bias=False):
        l = self.linear_layer(in_features, out_features, bias=bias)
        return l


    def _init_conv(self, conv):
        # Initialize with kaiming_normal, scale_fan=False, and mode='fan_in', nonlinearity="relu"
        nn.init.kaiming_normal_(
            conv.weight, mode="fan_in", nonlinearity="relu"
        )


def get_builder(args):
    if args.prune_algorithm is None or args.no_prune_conv:
        conv_layer = utils.dense.DenseConv
        deconv_layer = utils.dense.DenseConvTranspose
    else:
        if args.layerwise:
            conv_layer = utils.gmp.GMPConv
            deconv_layer = utils.gmp.GMPConvTranspose
        else:
            conv_layer = utils.gmp.GlobalGMPConv
            deconv_layer = utils.gmp.GlobalGMPConvTranspose

    if args.variable_rate:
        if args.prune_algorithm is None or args.no_prune_film:
            linear_layer = utils.dense.DenseLinear
        else:
            if args.layerwise:
                linear_layer = utils.gmp.GMPLinear
            else:
                linear_layer = utils.gmp.GlobalGMPLinear
    else:
        linear_layer = None
    
    print("==> Conv Layer: {}".format(conv_layer))
    print("==> Deconv Layer: {}".format(deconv_layer))
    print("==> Linear Layer: {}\n".format(linear_layer))

    builder = Builder(conv_layer, deconv_layer, linear_layer)

    return builder
