import os
import numpy as np
from glob import glob
import imageio

import PIL
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks


class MultiTaskDataset(Dataset):
    """MultiTaskDataset."""

    def __init__(self, dataset, data_dir, image_mean, data_list_1, data_list_2, output_size,
                 color_jitter, random_scale, random_mirror, random_crop, ignore_label):
        """
        Initialise an Multitask Dataloader.

        :param data_dir: path to the directory with images and masks.
        :param data_list_1: path to the file with lines of the form '/path/to/image /path/to/mask'.
        :param data_list_2: path to the file with lines of the form '/path/to/image /path/to/mask'.
        :param output_size: a tuple with (height, width) values, to which all the images will be resized to.
        :param random_scale: whether to randomly scale the images.
        :param random_mirror: whether to randomly mirror the images.
        :param random_crop: whether to randomly crop the images.
        :param ignore_label: index of label to ignore during the training.
        """
        assert dataset == 'nyu_v2'
        self.dataset = dataset
        self.data_dir = data_dir
        self.image_mean = np.load(image_mean)
        self.data_list_1 = data_list_1
        self.data_list_2 = data_list_2
        self.output_size = output_size

        self.color_jitter = None
        if color_jitter:
            print("Using color jitter")
            self.color_jitter = transforms.ColorJitter(hue=.05, saturation=.05)
        self.random_scale = random_scale
        self.random_mirror = random_mirror
        self.random_crop = random_crop

        self.ignore_label = ignore_label

        image_list_1, self.label_list_1 = read_labeled_image_list(self.data_dir, self.data_list_1)
        image_list_2, self.label_list_2 = read_labeled_image_list(self.data_dir, self.data_list_2)
        assert (image_list_1 == image_list_2)
        self.image_list = image_list_1

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(self.image_mean, (1., 1., 1.))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        label_1 = Image.open(self.label_list_1[idx])
        label_2 = Image.open(self.label_list_2[idx])
        w, h = image.size
        
        if self.color_jitter:
            image = self.color_jitter(image)

        if self.random_scale:
            scale = int(min(w, h) * (np.random.uniform() + 0.5))
            resize_bl = transforms.Resize(size=scale, interpolation=PIL.Image.BILINEAR)
            resize_nn = transforms.Resize(size=scale, interpolation=PIL.Image.NEAREST)
            image = resize_bl(image)
            label_1 = resize_nn(label_1)
            label_2 = resize_nn(label_2)

        if self.random_mirror:
            if np.random.uniform() < 0.5:
                image = TF.hflip(image)
                label_1 = TF.hflip(label_1)
                label_2 = TF.hflip(label_2)

        if self.random_crop:
            # pad the width if needed
            if image.size[0] < self.output_size[1]:
                image = TF.pad(image, (self.output_size[1] - image.size[0], 0))
                label_1 = TF.pad(label_1, (self.output_size[1] - label_1.size[0], 0), self.ignore_label, 'constant')
                label_2 = TF.pad(label_2, (self.output_size[1] - label_2.size[0], 0),
                                 tuple([self.ignore_label] * 3), 'constant')
            # pad the height if needed
            if image.size[1] < self.output_size[0]:
                image = TF.pad(image, (0, self.output_size[0] - image.size[1]))
                label_1 = TF.pad(label_1, (0, self.output_size[0] - label_1.size[1]), self.ignore_label, 'constant')
                label_2 = TF.pad(label_2, (0, self.output_size[0] - label_2.size[1]),
                                 tuple([self.ignore_label] * 3), 'constant')

            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.output_size)
            image = TF.crop(image, i, j, h, w)
            label_1 = TF.crop(label_1, i, j, h, w)
            label_2 = TF.crop(label_2, i, j, h, w)

        image = self.normalize(self.to_tensor(np.array(image) - 255.).float() + 255.)
        label_1 = self.to_tensor(np.array(label_1) - 255.) + 255.
        label_2 = self.to_tensor(np.array(label_2) - 255.) + 255.

        return image, label_1.long(), label_2.float()
