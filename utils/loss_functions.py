import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceLoss_BCE(nn.Module):
    def __init__(self, weight=None, alpha = 0.5, size_average=True):
        super(DiceLoss_BCE, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets, smooth=1):
        bce_loss = nn.BCEWithLogitsLoss()
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce = bce_loss(inputs, targets)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return self.alpha*bce + self.alpha*(1-dice)

class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(JaccardLoss, self).__init__()


    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()

        jaccard = (intersection + smooth)/(inputs.sum()+targets.sum()-intersection + smooth)
        return  1-jaccard

class JaccardLoss_BCE(nn.Module):
    def __init__(self, weight=None, alpha = 0.5, size_average=True):
        super(JaccardLoss_BCE, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets, bce_loss = nn.BCEWithLogitsLoss(), smooth=1):
        
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce = bce_loss(inputs, targets)
        intersection = (inputs * targets).sum()

        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        jaccard = (intersection + smooth)/(inputs.sum()+targets.sum()-intersection + smooth)
        return self.alpha*bce + self.alpha*(1-jaccard)

class ProposedLoss(nn.Module):
    def __init__(self, weight = None, alpha = 0.5, size_average = True):
        super(ProposedLoss,self).__init__()
        self.alpha = alpha
    def forward(self, inputs, targets,smooth = 1):
        bce_loss = torch.nn.CrossEntropyLoss()
        inputs = torch.nn.Softmax(dim=1)(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce = bce_loss(inputs, targets)
        intersection = (inputs * targets).sum()

        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        jaccard = (intersection + smooth)/(inputs.sum()+targets.sum()-intersection + smooth)
        return self.alpha*bce + (1-self.alpha)*0.5*(1-jaccard) + (1-self.alpha)*0.5*(1-dice)