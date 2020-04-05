import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from .utils.losses import seg_loss, normal_loss
from .utils.metrics import compute_hist, compute_angle, count_correct
from .utils.visualization import process_seg_label, process_normal_label


def get_tasks(cfg):
    if cfg.TASK == 'pixel':
        if cfg.DATASET == 'nyu_v2':
            return SegTask(cfg), NormalTask(cfg)
    

class Task:
    
    def loss(self, prediction, gt):
        raise NotImplementedError
    
    def log_visualize(self, prediction, gt, loss, writer, steps):
        raise NotImplementedError
        
    def accumulate_metric(self, prediction, gt, accumulator, distributed=False):
        raise NotImplementedError
        
    def aggregate_metric(self, accumulator):
        raise NotImplementedError

    
class SegTask(Task):
    
    def __init__(self, cfg):
        self.type = 'pixel'
        self.cfg = cfg
        
    def loss(self, prediction, gt):
        return seg_loss(prediction, gt, 255)
    
    def log_visualize(self, prediction, gt, loss, writer, steps):
        writer.add_scalar('loss/seg', loss.data.item(), steps)
        seg_pred, seg_gt = process_seg_label(prediction, gt, self.cfg.MODEL.NET1_CLASSES)
        writer.add_image('seg/pred', seg_pred, steps)
        writer.add_image('seg/gt', seg_gt, steps)

    def accumulate_metric(self, prediction, gt, accumulator, distributed=False):
        hist, correct_pixels, valid_pixels = compute_hist(prediction, gt, self.cfg.MODEL.NET1_CLASSES, 255)

        if distributed:  # gather metric results
            hist = torch.tensor(hist).cuda()
            correct_pixels = torch.tensor(correct_pixels).cuda()
            valid_pixels = torch.tensor(valid_pixels).cuda()

            # aggregate result to rank 0
            dist.reduce(hist, 0, dist.ReduceOp.SUM)
            dist.reduce(correct_pixels, 0, dist.ReduceOp.SUM)
            dist.reduce(valid_pixels, 0, dist.ReduceOp.SUM)

            hist = hist.cpu().numpy()
            correct_pixels = correct_pixels.cpu().item()
            valid_pixels = valid_pixels.cpu().item()

        accumulator['total_hist'] = accumulator.get('total_hist', 0.) + hist
        accumulator['total_correct_pixels'] = accumulator.get('total_correct_pixels', 0.) + correct_pixels
        accumulator['total_valid_pixels'] = accumulator.get('total_valid_pixels', 0.) + valid_pixels
        return accumulator
    
    def aggregate_metric(self, accumulator):
        total_hist = accumulator['total_hist']
        total_correct_pixels = accumulator['total_correct_pixels']
        total_valid_pixels = accumulator['total_valid_pixels']
        IoUs = np.diag(total_hist) / (np.sum(total_hist, axis=0) + np.sum(total_hist, axis=1) - np.diag(total_hist) + 1e-16)
        mIoU = np.mean(IoUs)
        pixel_acc = total_correct_pixels / total_valid_pixels
        return {
            'Mean IoU': mIoU,
            'Pixel Acc': pixel_acc
        }
        
        
class NormalTask(Task):
    
    def __init__(self, cfg):
        self.type = 'pixel'
        self.cfg = cfg
        
    def loss(self, prediction, gt):
        return normal_loss(prediction, gt, 255)
    
    def log_visualize(self, prediction, gt, loss, writer, steps):
        writer.add_scalar('loss/normal', loss.data.item(), steps)
        normal_pred, normal_gt = process_normal_label(prediction, gt, 255)
        writer.add_image('normal/pred', normal_pred, steps)
        writer.add_image('normal/gt', normal_gt, steps)
        
    def accumulate_metric(self, prediction, gt, accumulator, distributed=False):
        if distributed:
            prediction_list = [torch.zeros_like(prediction).cuda() for _ in range(dist.get_world_size())]
            gt_list = [torch.zeros_like(gt).cuda() for _ in range(dist.get_world_size())]
            dist.all_gather(prediction_list, prediction)
            dist.all_gather(gt_list, gt)

            if local_rank == 0:
                prediction = torch.cat(prediction_list, dim=0)
                gt = torch.cat(gt_list, dim=0)

        angle = compute_angle(prediction, gt, 255)
        angles = accumulator.get('angles', [])
        angles.append(angle)
        accumulator['angles'] = angles
        return accumulator
    
    def aggregate_metric(self, accumulator):
        angles = accumulator['angles']
        angles = np.concatenate(angles, axis=0)
        return {
            'Mean': np.mean(angles),
            'Median': np.median(angles),
            'RMSE': np.sqrt(np.mean(angles ** 2)),
            '11.25': np.mean(np.less_equal(angles, 11.25)) * 100,
            '22.5': np.mean(np.less_equal(angles, 22.5)) * 100,
            '30': np.mean(np.less_equal(angles, 30.0)) * 100,
            '45': np.mean(np.less_equal(angles, 45.0)) * 100
        }
