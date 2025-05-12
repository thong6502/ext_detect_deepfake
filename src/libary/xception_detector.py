import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from .xception import Xception
import yaml
from .base_metrics_class import calculate_metrics_for_train



class XceptionDetector(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.backbone = self.build_backbone(num_classes)
        self.loss_func = self.build_loss()
        
    def build_backbone(self, num_classes):
        # prepare the backbone
        backbone = Xception(num_classes)
        return backbone
    
    def build_loss(self):
        # prepare the loss function
        loss_func = CrossEntropyLoss()
        return loss_func
    
    def features(self, input) -> torch.tensor:
        return self.backbone.features(input) #32,3,256,256

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        overall_loss = loss
        loss_dict = {'overall': overall_loss, 'cls': loss,}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict) -> dict:
        # get the features by backbone
        features = self.features(data_dict['image'])
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict
