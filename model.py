from torchvision.models.resnet import resnet101, ResNet, Bottleneck, model_urls
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torchvision
from torchvision.datasets import VOCDetection
import PIL
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import dataset_utils as dt
import config
import rfcn_utils as rfcn


class Network(nn.Module):

    def __init__(self, load_resnet=True):
        super(Network, self).__init__()
        self.backbone = Backbone(pretrained=load_resnet)
        self.conv1 = nn.Conv2d(2048, 1024, (1, 1))
        self.conv_score_maps = nn.Conv2d(1024,
                                         config.SM_BIN_LENGTH * config.SM_BIN_LENGTH * (config.NUM_CLASSES + 1),
                                         1)  # Alert paper didn't mention size
        self.relu = nn.ReLU()
        self.RPN = RPN()

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1(x)
        x = self.relu(x)

        # cls_score, bbox_pred = self.RPN(x)
        # TODO filter boxes

        x = self.conv_score_maps(x)
        scores = rfcn.map_roi_to_class_probs(0, 0, 3, 3, x.squeeze(0))


        return x


class RPN(nn.Module):

    def __init__(self):
        super(RPN, self).__init__()
        self.conv1 = nn.Conv2d(1024, 512, 3)
        self.relu = nn.ReLU()
        self.conv_cls = nn.Conv2d(512, (config.NUM_CLASSES + 1) * config.NUM_RPN_ANCHORS, 1)
        self.conv_bbox = nn.Conv2d(512, 4 * config.NUM_RPN_ANCHORS, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        cls_score = self.conv_cls(x)
        cls_score = cls_score.reshape(cls_score.shape[0],
                                      cls_score.shape[2],
                                      cls_score.shape[3],
                                      config.NUM_RPN_ANCHORS,
                                      config.NUM_CLASSES + 1)

        bbox_pred = self.conv_bbox(x)
        bbox_pred = bbox_pred.reshape(bbox_pred.shape[0],
                                      bbox_pred.shape[2],
                                      bbox_pred.shape[3],
                                      config.NUM_RPN_ANCHORS, 4)

        return cls_score, bbox_pred


# Resnet 101 but with the avg pooling and fc layers
class Backbone(ResNet):

    def __init__(self, pretrained=True):
        super(Backbone, self).__init__(Bottleneck, [3, 4, 23, 3])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))  # Alert does this load correctly?

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


if __name__ == "__main__":
    dataloader = dt.get_data_loader()
    model = Network(load_resnet=False)
    for batch in dataloader:
        images_batch, annotation_batch = batch[0], batch[1]

        for img in images_batch:
            yhat = model(img.unsqueeze(0))
            break

        break


