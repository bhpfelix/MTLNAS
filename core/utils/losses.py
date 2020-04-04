import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli


# poly schedule
def poly(start, end, steps, total_steps, period, power):
    """
    Default goes from start to end
    """
    delta = end - start
    rate = float(steps) / total_steps
    if rate <= period[0]:
        return start
    elif rate >= period[1]:
        return end
    base = total_steps * period[0]
    ceil = total_steps * period[1]
    return end - delta * (1. - float(steps - base) / (ceil - base)) ** power
    
    
# Build Losses
def logits_loss(prediction, gt, reduce):
    reduction = 'mean' if reduce else 'none'
    prediction = F.softmax(prediction, dim=1)
    return F.mse_loss(prediction, gt, reduction=reduction)


def class_loss(prediction, gt):
    return F.cross_entropy(prediction, gt)

    
def depth_loss(prediction, gt, ignore_label=255):
    N, C, H, W = gt.size()
    prediction = F.interpolate(prediction, (H, W), mode='bilinear', align_corners=True)
    prediction = prediction.view(-1)
    gt = gt.view(-1)
    mask = (gt != ignore_label).nonzero()
    prediction = prediction[mask]
    gt = gt[mask]
    loss = torch.sqrt(torch.mean((prediction - gt)**2))
    return loss


def normal_loss(prediction, gt, ignore_label=255):
    '''Compute normal loss. (normalized cosine distance)
    Args:
      prediction: the output of cnn. Float type
      gt: the groundtruth. Float type
    '''
    N, C, H, W = gt.size()
    prediction = F.interpolate(prediction, (H, W), mode='bilinear', align_corners=True)
    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    gt = gt.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    mask = ((gt == ignore_label).sum(dim=1) - 3).nonzero().squeeze()
    prediction = prediction[mask]
    gt = gt[mask]
    loss = F.cosine_similarity(gt, prediction)
    return 1 - loss.mean()


def seg_loss(prediction, gt, ignore_label=255):
    N, C, H, W = gt.size()
    prediction = F.interpolate(prediction, (H, W), mode='bilinear', align_corners=True)
    loss = F.cross_entropy(prediction, gt.squeeze(1), ignore_index=ignore_label)
    return loss


def entropy_loss(arch_params):
    loss = []
    for arch_param in arch_params:
        probs = Bernoulli(logits=arch_param)
        loss.append(probs.entropy().mean())
    loss = torch.mean(torch.stack(loss))
    return loss


def l1_loss(arch_params, weights=None):
    loss = []
    for arch_param in arch_params:
        cost = torch.sigmoid(arch_param)
        if weights is not None:
            cost = cost * weights
        loss.append(cost.mean())
    loss = torch.mean(torch.stack(loss))
    return loss
