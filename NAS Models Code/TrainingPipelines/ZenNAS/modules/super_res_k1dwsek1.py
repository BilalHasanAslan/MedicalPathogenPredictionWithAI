# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import copy
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .blocks_basic import STD_BITS_LUT, BaseSuperBlock, ConvKXBN, SE


class ResK1DWSEK1(nn.Module):

    def __init__(self,
                 structure_info,
                 no_create=False,
                 dropout_channel=None,
                 dropout_layer=None,
                 **kwargs):
        '''

        :param structure_info: {
            'class': 'ResK1DWSEK1',
            'in': in_channels,
            'out': out_channels,
            's': stride (default=1),
            'k': kernel_size,
            'p': padding (default=(k-1)//2,
            'g': grouping (default=1),
            'btn': bottleneck_channels,
            'nbitsA': input activation quant nbits list(default=8),
            'nbitsW': weight quant nbits list(default=8),
            'act': activation (default=relu),
        }
        :param NAS_mode:
        '''

        super().__init__()

        #if 'class' in structure_info:
        #    assert structure_info['class'] == self.__class__.__name__

        self.in_channels = structure_info['in']
        self.out_channels = structure_info['out']
        self.kernel_size = structure_info['k']
        self.stride = 1 if 's' not in structure_info else structure_info['s']
        self.bottleneck_channels = structure_info['btn']
        assert self.stride == 1 or self.stride == 2
        if 'act' not in structure_info:
            self.activation_function = nn.SiLU()
        else:
            self.activation_function = get_activation(structure_info['act'])
        self.no_create = no_create
        self.dropout_channel = dropout_channel
        self.dropout_layer = dropout_layer

        if 'force_resproj' in structure_info:
            self.force_resproj = structure_info['force_resproj']
        else:
            self.force_resproj = False

        if 'nbitsA' in structure_info and 'nbitsW' in structure_info:
            self.quant = True
            self.nbitsA = structure_info['nbitsA']
            self.nbitsW = structure_info['nbitsW']
            if len(self.nbitsA) != 3 or len(self.nbitsW) != 3:
                raise ValueError(
                    'nbitsA/W must has three elements in %s, not nbitsA %d or nbitsW %d'
                    % (self.__class__, len(self.nbitsA), len(self.nbitsW)))

        else:
            self.quant = False

        if 'g' in structure_info:
            self.groups = structure_info['g']
        else:
            self.groups = 1

        if 'p' in structure_info:
            self.padding = structure_info['p']
        else:
            self.padding = (self.kernel_size - 1) // 2

        self.model_size = 0.0
        self.flops = 0.0

        self.block_list = []

        conv1_info = {
            'in': self.in_channels,
            'out': self.bottleneck_channels,
            'k': 1,
            's': 1,
            'g': self.groups,
            'p': 0
        }
        conv2_info = {
            'in': self.bottleneck_channels,
            'out': self.bottleneck_channels,
            'k': self.kernel_size,
            's': self.stride,
            'g': self.bottleneck_channels,
            'p': self.padding
        }
        se2_info = {
            'in': self.bottleneck_channels,
            'out': self.bottleneck_channels
        }
        conv3_info = {
            'in': self.bottleneck_channels,
            'out': self.out_channels,
            'k': 1,
            's': 1,
            'g': self.groups,
            'p': 0
        }
        if self.quant:
            conv1_info = {
                **conv1_info,
                **{
                    'nbitsA': self.nbitsA[0],
                    'nbitsW': self.nbitsW[0]
                }
            }
            conv2_info = {
                **conv2_info,
                **{
                    'nbitsA': self.nbitsA[1],
                    'nbitsW': self.nbitsW[1]
                }
            }
            conv3_info = {
                **conv3_info,
                **{
                    'nbitsA': self.nbitsA[2],
                    'nbitsW': self.nbitsW[2]
                }
            }

        self.conv1 = ConvKXBN(conv1_info, no_create=no_create, **kwargs)
        self.conv2 = ConvKXBN(conv2_info, no_create=no_create, **kwargs)
        self.se2 = SE(se2_info, no_create=no_create, **kwargs)
        self.conv3 = ConvKXBN(conv3_info, no_create=no_create, **kwargs)

        # if self.no_create:
        #     pass
        # else:
        #     network_weight_stupid_bn_zero_init(self.conv3)

        self.block_list.append(self.conv1)
        self.block_list.append(self.conv2)
        self.block_list.append(self.se2)
        self.block_list.append(self.conv3)

        # residual link
        self.is_reslink = True
        if self.in_channels == self.out_channels:
            if self.no_create:
                pass
            else:
                self.residual_proj = nn.Identity()

        elif self.force_resproj:
            if self.quant:
                self.residual_proj = ConvKXBN(
                    {
                        'in': self.in_channels,
                        'out': self.out_channels,
                        'k': 1,
                        's': 1,
                        'g': 1,
                        'p': 0,
                        'nbitsA': self.nbitsA[0],
                        'nbitsW': self.nbitsW[0]
                    },
                    no_create=no_create)
            else:
                self.residual_proj = ConvKXBN(
                    {
                        'in': self.in_channels,
                        'out': self.out_channels,
                        'k': 1,
                        's': 1,
                        'g': 1,
                        'p': 0
                    },
                    no_create=no_create)
        else:
            self.is_reslink = False

        if self.is_reslink:
            if self.stride == 2:
                if self.no_create:
                    pass
                else:
                    self.residual_downsample = nn.AvgPool2d(
                        kernel_size=2, stride=2)
                self.flops += self.in_channels
            else:
                if self.no_create:
                    pass
                else:
                    self.residual_downsample = nn.Identity()

    def forward(self, x, compute_reslink=True):
        if self.is_reslink:
            reslink = self.residual_downsample(x)
            reslink = self.residual_proj(reslink)

        output = x
        output = self.conv1(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)
        output = self.activation_function(output)
        output = self.conv2(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)
        output = self.activation_function(output)
        output = self.se2(output)
        output = self.conv3(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)

        if self.dropout_layer is not None:
            if np.random.rand() <= self.dropout_layer:
                output = 0 * output
            else:
                output = output
        if self.is_reslink:
            output = output + reslink

        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)

        output = self.activation_function(output)

        return output

class SuperResK1DWSEK1(BaseSuperBlock):

    def __init__(self,
                 structure_info,
                 no_create=False,
                 dropout_channel=None,
                 dropout_layer=None,
                 **kwargs):
        '''

        :param structure_info: {
            'class': 'SuperResK1DWSEK1',
            'in': in_channels,
            'out': out_channels,
            's': stride (default=1),
            'k': kernel_size,
            'p': padding (default=(k-1)//2,
            'g': grouping (default=1),
            'btn':, bottleneck_channels,
            'L': num_inner_layers,
        }
        :param NAS_mode:
        '''
        structure_info['inner_class'] = 'ResK1DWSEK1'
        super().__init__(
            structure_info=structure_info,
            no_create=no_create,
            inner_class=ResK1DWSEK1,
            dropout_channel=dropout_channel,
            dropout_layer=dropout_layer,
            **kwargs)


__module_blocks__ = {
    'ResK1DWSEK1': ResK1DWSEK1,
    'SuperResK1DWSEK1': SuperResK1DWSEK1,
}
