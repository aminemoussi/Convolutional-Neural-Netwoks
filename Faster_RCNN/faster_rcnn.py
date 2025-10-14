import math
from os import pread

import anchor_handling
import core
import region_proposal_network
import roi_head
import torch
import torch.nn as nn
import torchvision
from torch._C import dtype
from torch.cuda import _compile_kernel


class faster_rcnn(nn.Module):
    def __init__(self, model_config, num_classes=21):
        super(faster_rcnn, self).__init__()
        self.model_config = model_config
        vgg16 = torchvision.models.vgg16(pretrained=True)
        # vgg16 except last classification layer
        self.backbone = vgg16.features[:-1]
        self.rpn = region_proposal_network.RegionProposalNetwork(in_channels=512)
        self.roi_head = roi_head.ROIHead(
            model_config, num_classes=num_classes, in_channels=512
        )

        # freeze firs 10 layers
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = 600
        self.max_size = 1000
