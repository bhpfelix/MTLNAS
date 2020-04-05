import numpy as np
import torch
import torch.nn.functional as F

    
def compute_hist(prediction, gt, n_classes, ignore_label):
    N, C, H, W = gt.size()
    prediction = F.interpolate(prediction, (H, W), mode='bilinear', align_corners=True)
    prediction = torch.argmax(prediction, dim=1).flatten().cpu().numpy()
    gt = gt.flatten().cpu().numpy()
    keep = np.logical_not(gt == ignore_label)
    merge = prediction[keep] * n_classes + gt[keep]
    hist = np.bincount(merge, minlength=n_classes**2)
    hist = hist.reshape((n_classes, n_classes))
    correct_pixels = np.diag(hist).sum()
    valid_pixels = keep.sum()
    return hist, correct_pixels, valid_pixels


def compute_angle(prediction, gt, ignore_label):
    N, C, H, W = gt.size()
    prediction = F.interpolate(prediction, (H, W), mode='bilinear', align_corners=True)
    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    gt = gt.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    mask = ((gt == ignore_label).sum(dim=1) - 3).nonzero().squeeze()
    prediction = prediction[mask]
    gt = gt[mask]
    cosine_distance = F.cosine_similarity(gt, prediction)
    cosine_distance = cosine_distance.cpu().numpy()
    cosine_distance = np.minimum(np.maximum(cosine_distance, -1.0), 1.0)
    angles = np.arccos(cosine_distance) / np.pi * 180.0
    return angles


def count_correct(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res
    