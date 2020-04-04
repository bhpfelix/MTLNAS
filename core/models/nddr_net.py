import numpy as np
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.common_layers import batch_norm, get_nddr
from core.tasks import get_tasks
from core.utils import AttrDict
from core.utils.losses import poly


class SingleTaskNet(nn.Module):
    def __init__(self, cfg, net1, net2):
        super(SingleTaskNet, self).__init__()
        self.cfg = cfg
        self.net1 = net1
        self.net2 = net2
        assert len(net1.stages) == len(net2.stages)
        self.task1, self.task2 = get_tasks(cfg)

        self.num_stages = len(net1.stages)
        self._step = 0
        
    def step(self):
        self._step += 1
        
    def loss(self, x, labels):
        label_1, label_2 = labels
        result = self.forward(x)
        result.loss1 = self.task1.loss(result.out1, label_1)
        result.loss2 = self.task2.loss(result.out2, label_2)
        result.loss = result.loss1 + self.cfg.TRAIN.TASK2_FACTOR * result.loss2
        return result

    def forward(self, x):
        N, C, H, W = x.size()
        y = x.clone()
        x = self.net1.base(x)
        y = self.net2.base(y)
        for stage_id in range(self.num_stages):
            x = self.net1.stages[stage_id](x)
            y = self.net2.stages[stage_id](y)
        x = self.net1.head(x)
        y = self.net2.head(y)
        return AttrDict({'out1': x, 'out2': y})
    
    
class SharedFeatureNet(nn.Module):
    def __init__(self, cfg, net1, net2):
        super(SharedFeatureNet, self).__init__()
        self.cfg = cfg
        self.net1 = net1
        self.net2 = net2
        assert len(net1.stages) == len(net2.stages)
        self.task1, self.task2 = get_tasks(cfg)

        self.num_stages = len(net1.stages)
        self._step = 0
        
    def step(self):
        self._step += 1
        
    def loss(self, x, labels):
        label_1, label_2 = labels
        result = self.forward(x)
        result.loss1 = self.task1.loss(result.out1, label_1)
        result.loss2 = self.task2.loss(result.out2, label_2)
        result.loss = result.loss1 + self.cfg.TRAIN.TASK2_FACTOR * result.loss2
        return result

    def forward(self, x):
        x = self.net1.base(x)
        for stage_id in range(self.num_stages):
            x = self.net1.stages[stage_id](x)
        out1 = self.net1.head(x)
        out2 = self.net2.head(x)
        return AttrDict({'out1': out1, 'out2': out2})
    

class NDDRNet(nn.Module):
    def __init__(self, cfg, net1, net2):
        super(NDDRNet, self).__init__()
        self.cfg = cfg
        self.net1 = net1
        self.net2 = net2
        assert len(net1.stages) == len(net2.stages)
        self.task1, self.task2 = get_tasks(cfg)

        self.num_stages = len(net1.stages)
        nddrs = []
        total_channels = 0
        for stage_id in range(self.num_stages):
            out_channels = net1.stages[stage_id].out_channels
            assert out_channels == net2.stages[stage_id].out_channels
            if stage_id in cfg.TRAIN.AUX_LAYERS:
                total_channels += out_channels
            nddr = get_nddr(cfg, out_channels, out_channels)
            nddrs.append(nddr)
        nddrs = nn.ModuleList(nddrs)
        
        self.aux = cfg.TRAIN.AUX
        if self.aux:
            print("Using shortcut")
            self.aux_conv1 = nn.Sequential(
                nn.Conv2d(total_channels, 256, kernel_size=3, padding=1, bias=False),
                batch_norm(256, eps=1e-03, momentum=cfg.MODEL.BATCH_NORM_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                nn.Conv2d(256, cfg.MODEL.NET1_CLASSES, kernel_size=1)
            )
            self.aux_conv2 = nn.Sequential(
                nn.Conv2d(total_channels, 256, kernel_size=3, padding=1, bias=False),
                batch_norm(256, eps=1e-03, momentum=cfg.MODEL.BATCH_NORM_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                nn.Conv2d(256, cfg.MODEL.NET2_CLASSES, kernel_size=1)
            )

        self.nddrs = nn.ModuleDict({
            'nddrs': nddrs,
        })
        
        self._step = 0
        
    def step(self):
        self._step += 1
        
    def loss(self, x, labels):
        label_1, label_2 = labels
        result = self.forward(x)
        result.loss1 = self.task1.loss(result.out1, label_1)
        result.loss2 = self.task2.loss(result.out2, label_2)
        result.loss = result.loss1 + self.cfg.TRAIN.TASK2_FACTOR * result.loss2
        if self.aux:
            result.aux_loss1 = self.task1.loss(result.aux1, label_1)
            result.aux_loss2 = self.task2.loss(result.aux2, label_2)
            result.aux_loss = result.aux_loss1 + self.cfg.TRAIN.TASK2_FACTOR * result.aux_loss2
            result.aux_weight = poly(start=self.cfg.TRAIN.AUX_WEIGHT, end=0.,
                                    steps=self._step, total_steps=self.cfg.TRAIN.STEPS,
                                    period=self.cfg.TRAIN.AUX_PERIOD,
                                    power=1.)
            result.loss += result.aux_weight * result.aux_loss
        return result

    def forward(self, x):
        N, C, H, W = x.size()
        y = x.clone()
        x = self.net1.base(x)
        y = self.net2.base(y)
        xs, ys = [], []
        for stage_id in range(self.num_stages):
            x = self.net1.stages[stage_id](x)
            y = self.net2.stages[stage_id](y)
            if isinstance(x, list):
                x[0], y[0] = self.nddrs['nddrs'][stage_id](x[0], y[0])
            else:
                x, y = self.nddrs['nddrs'][stage_id](x, y)
            if self.aux and self.training and stage_id in self.cfg.TRAIN.AUX_LAYERS:
                xs.append(x)
                ys.append(y)
        x = self.net1.head(x)
        y = self.net2.head(y)
        result = AttrDict({'out1': x, 'out2': y})
        
        if self.aux and self.training:
            _, _, h, w = x.size()
            aux_x = torch.cat([F.interpolate(_x, (h, w), mode='bilinear', align_corners=True) for _x in xs[:-1]] + [xs[-1]],
                          dim=1)
            aux_y = torch.cat([F.interpolate(_y, (h, w), mode='bilinear', align_corners=True) for _y in ys[:-1]] + [ys[-1]],
                          dim=1)
            result.aux1 = self.aux_conv1(aux_x)
            result.aux2 = self.aux_conv2(aux_y)
        return result
