import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/yxdr/pytorch-multi-class-focal-loss/blob/master/FocalLoss.py
class MultiFocalLoss(nn.Module):
#class MultiFocalLoss:
    def __init__(self, alpha_t=None, gamma=0, size_average=False):
        """
        :param alpha_t:A List of weights for each class
        :param gamma: scalar
        """
        super(MultiFocalLoss,self).__init__()
        self.alpha_t = torch.tensor(alpha_t) if alpha_t else None
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, outputs, targets):
    #def __call__(self, outputs, targets):
        if self.alpha_t is not None and self.gamma == 0:
            # focal_loss = torch.nn.functional.cross_entropy(outputs, targets,weight=self.alpha_t, reduction='none')
            focal_loss = F.cross_entropy(outputs, targets, weight=self.alpha_t, reduction='none')

        if self.alpha_t is not None and self.gamma != 0:
            ce_loss = F.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            ce_loss = F.cross_entropy(outputs, targets, weight=self.alpha_t, reduction='none')

            focal_loss = ((1 - p_t) ** self.gamma * ce_loss)  # mean over the batch
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


""""""
if __name__ == '__main__':
    # outputs = torch.tensor([[2, 2, 1], [2.5, 1, 1]])
    # outputs = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.5, 0.2]])
    outputs = torch.tensor([[0.1, 0.7, 0.2], [0.03, 0.8, 0.17]])
    # print(outputs.shape)
    targets = torch.tensor([1, 0])
    focal1 = MultiFocalLoss(alpha_t=[0.3, 0.1, 0.8], gamma=0)
    ce = focal1(outputs, targets)
    print("ce:", ce)
    # floss = MultiFocalLoss(alpha_t=[0.3, 0.2, 0.5], gamma=2)
    floss2 = MultiFocalLoss(alpha_t=[0.3, 0.2, 0.5], gamma=2)
    cal = floss2(outputs, targets)
    print(cal)

