# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf


class MultiCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2,local_transform=None, num_crops=8, ):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
        self.local_transform = local_transform
        self.num_crops = num_crops

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        local_crops = []
        for _ in range(self.num_crops):
            local_crops.append(self.local_transform(x))
            
        return [im1, im2] + local_crops


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)