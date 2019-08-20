import torch
from torch import nn
from torch.nn import MultiLabelSoftMarginLoss


def criterion_default():
    return nn.CrossEntropyLoss(reduction='mean')


def criterion_BCELoss():
    return nn.BCELoss()


def criterion_BCEWithLogitsLoss():
    return nn.BCEWithLogitsLoss()


def criterion_MSELoss():
    return nn.MSELoss()


def criterion_KLDivLoss():
    return nn.KLDivLoss()

def criterion_NLLLoss():
    return nn.NLLLoss()


global_first_time = True


class VectorCELoss(nn.Module):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """

    def __init__(self, size_average=True):
        super().__init__()
        self.logsoftmax = nn.LogSoftmax()
        self.size_average = size_average

    def forward(self, output, target):
        if self.size_average:
            return torch.mean(torch.sum(-target * self.logsoftmax(output), dim=1))
        else:
            return torch.sum(torch.sum(-target * self.logsoftmax(output), dim=1))


class MultiVectorCELoss(nn.Module):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """

    def __init__(self, size_average=True):
        super().__init__()
        self.logsoftmax = nn.LogSoftmax()
        self.size_average = size_average

    def forward(self, output, target):
        totalloss = 0
        for t, o in zip(target, output):
            ts = (t > 0.5).nonzero()
            con = (t <= 0.5)
            newo = o[con]
            newt = t[con]
            batch_loss = 0
            for tempt in ts:
                batch_loss = batch_loss + torch.sum(
                    -torch.cat([newt, t[tempt]]) * torch.nn.functional.log_softmax(torch.cat([newo, o[tempt]]), dim=0))
            if len(ts):
                totalloss = totalloss + batch_loss / len(ts)
            else:
                totalloss = totalloss + torch.sum(
                    -newt * torch.nn.functional.log_softmax(newo, dim=0))
        if self.size_average:
            return totalloss / output.size(0)
        else:
            return totalloss


first_LOSS_cut = 0



class MixUpLoss_CrossEntropyLoss(nn.Module):
    "Adapts the loss function to go with mixup."

    def __init__(self, crit=None):
        super().__init__()
        if crit is None:
            self.crit = nn.CrossEntropyLoss(reduction="none")
        else:
            self.crit = crit

    def forward(self, output, target1, target2=None, lmpas=None, mode=None):
        global global_first_time
        if target2 is None: return self.crit(output, target1).mean()
        if global_first_time: print("using mix up loss!! ", self.crit)
        global_first_time = False
        loss1, loss2 = self.crit(output, target1), self.crit(output, target2)
        return (loss1 * lmpas + loss2 * (1 - lmpas)).mean()


class MixUpLoss_multilabel(nn.Module):
    "Adapts the loss function to go with mixup."

    def __init__(self, crit=None):
        super().__init__()
        if crit is None:
            self.crit = nn.CrossEntropyLoss(reduction="none")
        else:
            self.crit = crit

    def forward(self, output, target1, target2=None, lmpas=None, mode=None):
        global global_first_time
        if target2 is None: return self.crit(output, target1).mean()
        if global_first_time: print("using mix up loss!! ", self.crit)
        global_first_time = False
        loss1, loss2 = self.crit(output, target1), self.crit(output, target2)
        lmpas = lmpas.unsqueeze(1)
        return (loss1 * lmpas + loss2 * (1 - lmpas)).mean()


class MixUpLoss_Targetmix(nn.Module):
    "Adapts the loss function to go with mixup."

    def __init__(self, crit=None):
        super().__init__()
        if crit is None:
            raise RuntimeError("Not a valid loss")
        else:
            self.crit = crit

    def forward(self, output, target1, target2=None, lmpas=None, mode=None):
        global global_first_time
        if target2 is None: return self.crit(output, target1).mean()
        if global_first_time: print("using targets mix up loss!! ", self.crit)
        global_first_time = False
        targ = target1 * lmpas.unsqueeze(1) + target2 * (1 - lmpas).unsqueeze(1)
        return self.crit(output, targ)


def criterion_MultiLabelSoftMarginLoss():
    return MultiLabelSoftMarginLoss(reduction="mean")


def criterion_mixup_default():
    return MixUpLoss_CrossEntropyLoss()


def criterion_mixup_BCEWithLogitsLoss():
    return MixUpLoss_CrossEntropyLoss(crit=nn.BCEWithLogitsLoss())


def criterion_mixup_MultiLabelSoftMarginLoss():
    return MixUpLoss_CrossEntropyLoss(crit=nn.MultiLabelSoftMarginLoss(reduction="none"))


def criterion_mixup_KLDivLoss():
    return MixUpLoss_multilabel(crit=nn.KLDivLoss(reduction="none"))


def criterion_VCELoss():
    return VectorCELoss()


def criterion_VCELosscut():
    return VectorCELossCut()


def criterion_mixup_vce():
    return MixUpLoss_CrossEntropyLoss(crit=VectorCELoss())


def criterion_targetsmixupCE():
    return MixUpLoss_Targetmix(crit=VectorCELoss())


def criterion_multitargetsmixupCE():
    return MixUpLoss_Targetmix(crit=MultiVectorCELoss())


def criterion_mixup_upweight():
    pos = torch.ones([80]) * 10
    pos = pos.cuda()
    return MixUpLoss_CrossEntropyLoss(crit=nn.BCEWithLogitsLoss(pos_weight=pos))


def criterion_bce_upweight():
    pos = torch.ones([80]) * 10
    pos = pos.cuda()
    return nn.BCEWithLogitsLoss(pos_weight=pos)


def criterion_kbce_upweight():
    return KBCE()

def criterion_single_upweight():
    pos = torch.ones([1]) * 2
    pos = pos.cuda()
    print("criterion_single_upweight 2!!!")
    return nn.BCEWithLogitsLoss(pos_weight=pos)



global_first_timeB = True

import torch.nn.functional as F



class EDMLoss(nn.Module):
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()


def criterion_emd():
    return EDMLoss()
