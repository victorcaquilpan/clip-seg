import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
from matplotlib import pyplot as plt
from torch import Tensor, einsum


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    #     return dice_loss_avg
    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        # Flatten predictions and targets
        prediction = predict.view(-1)
        target = target.view(-1)
        
        # Calculate intersection and union
        intersection = (prediction * target).sum()
        dice = (2. * intersection + self.smooth) / (prediction.sum() + target.sum() + self.smooth)
    
        # Dice loss is 1 - Dice score
        dice_loss = 1 - dice

        return dice_loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target, name, TEMPLATE):
        
        total_loss = []
        predict = F.sigmoid(predict)

        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            dataset_index = int(name[b][0:2])
            if dataset_index == 10:
                template_key = name[b][0:2] + '_' + name[b][17:19]
            elif dataset_index == 1:
                if int(name[b][-2:]) >= 60:
                    template_key = '01_2'
                else:
                    template_key = '01'
            else:
                template_key = name[b][0:2]
            organ_list = TEMPLATE[template_key]
            for organ in organ_list:
                dice_loss = self.dice(predict[b, organ-1], target[b, organ-1])
                total_loss.append(dice_loss)
            
        total_loss = torch.stack(total_loss)

        return total_loss.sum()/total_loss.shape[0]

        

class Multi_BCELoss(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(Multi_BCELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target, name, TEMPLATE):
        assert predict.shape[2:] == target.shape[2:], 'predict & target shape do not match'

        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            dataset_index = int(name[b][0:2])
            if dataset_index == 10:
                template_key = name[b][0:2] + '_' + name[b][17:19]
            elif dataset_index == 1:
                if int(name[b][-2:]) >= 60:
                    template_key = '01_2'
                else:
                    template_key = '01'
            else:
                template_key = name[b][0:2]
            organ_list = TEMPLATE[template_key]
            for organ in organ_list:
                ce_loss = self.criterion(predict[b, organ-1], target[b, organ-1])
                total_loss.append(ce_loss)
        total_loss = torch.stack(total_loss)

        # print(name, total_loss, total_loss.sum()/total_loss.shape[0])

        return total_loss.sum()/total_loss.shape[0]
