import numpy as np

from .supernet import GeneralizedMTLNASNet
from .nddr_net import SingleTaskNet, SharedFeatureNet, NDDRNet
from .vgg16_lfov_bn import DeepLabLargeFOVBN
from .vgg16_lfov_bn_16_stages import DeepLabLargeFOVBN16


def depth_limited_connectivity_matrix(stage_config, limit=3):
    """

    :param stage_config: list of number of layers in each stage
    :param limit: limit of depth difference between connected layers, pass in -1 to disable
    :return: connectivity matrix
    """
    network_depth = np.sum(stage_config)
    stage_depths = np.cumsum([0] + stage_config)
    matrix = np.zeros((network_depth, network_depth)).astype('int')
    for i in range(network_depth):
        j_limit = stage_depths[np.argmax(stage_depths > i) - 1]
        for j in range(network_depth):
            if j <= i and i - j < limit and j >= j_limit:
                matrix[i, j] = 1.
    return matrix


def vgg_connectivity():
    return depth_limited_connectivity_matrix([2, 2, 3, 3, 3])


def get_model(cfg, task1, task2):
    if cfg.TASK == 'pixel':
        if cfg.MODEL.BACKBONE == 'VGG16':
            net1 = DeepLabLargeFOVBN(3, cfg.MODEL.NET1_CLASSES, weights=cfg.TRAIN.WEIGHT_1)
            net2 = DeepLabLargeFOVBN(3, cfg.MODEL.NET2_CLASSES, weights=cfg.TRAIN.WEIGHT_2)
        elif cfg.MODEL.BACKBONE == 'VGG16_13_Stage':
            net1 = DeepLabLargeFOVBN16(3, cfg.MODEL.NET1_CLASSES, weights=cfg.TRAIN.WEIGHT_1)
            net2 = DeepLabLargeFOVBN16(3, cfg.MODEL.NET2_CLASSES, weights=cfg.TRAIN.WEIGHT_2)
        else:
            raise NotImplementedError
        

    if cfg.ARCH.SEARCHSPACE == 'GeneralizedMTLNAS':
        if cfg.MODEL.BACKBONE == 'VGG16_13_Stage':
            connectivity = vgg_connectivity
        else:
            raise NotImplementedError
        
        model = GeneralizedMTLNASNet(cfg, net1, net2,
                                     net1_connectivity_matrix=connectivity(),
                                     net2_connectivity_matrix=connectivity())
    else:
        if cfg.MODEL.SINGLETASK:
            print('Running Single Task Baseline')
            model = SingleTaskNet(cfg, net1, net2)
        elif cfg.MODEL.SHAREDFEATURE:
            print('Running Shared feature baseline')
            model = SharedFeatureNet(cfg, net1, net2)
        else:
            model = NDDRNet(cfg, net1, net2)
        
    return model
