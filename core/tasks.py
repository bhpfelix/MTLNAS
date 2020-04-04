import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from .utils.losses import logits_loss, class_loss, seg_loss, normal_loss, depth_loss
from .utils.metrics import squared_error, compute_hist, compute_angle, count_correct
from .utils.visualization import process_seg_label, process_normal_label, save_depth_label


def get_tasks(cfg):
    if cfg.TASK == 'pixel':
        if cfg.DATASET == 'gta':
            return SegTask(cfg), DepthTask(cfg)
        else:
            return SegTask(cfg), NormalTask(cfg)
    elif cfg.TASK == 'image':
        return ObjectClassTask(cfg), SceneClassTask(cfg)
    

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
    
    
class DepthTask(Task):
    
    def __init__(self, cfg):
        self.type = 'pixel'
        self.cfg = cfg
        
    def loss(self, prediction, gt):
        return depth_loss(prediction, gt, 255)
    
    def log_visualize(self, prediction, gt, loss, writer, steps):
        B, C, H, W = gt.size()
        prediction = F.interpolate(prediction, (H, W), mode='bilinear', align_corners=True)
        prediction = prediction[0].detach()
        gt = gt[0]
        prediction = prediction.squeeze().cpu().numpy()
        gt = gt.squeeze().cpu().numpy()
        gt[gt == 255.] = -1
    
        writer.add_scalar('loss/depth', loss.data.item(), steps)
        
        experiment_log_dir = os.path.join(self.cfg.LOG_DIR, self.cfg.EXPERIMENT_NAME, 'tmp')
        if not os.path.isdir(experiment_log_dir):
            os.makedirs(experiment_log_dir)
        prediction = save_depth_label(prediction, os.path.join(experiment_log_dir, "depth_pred.png"))
        gt = save_depth_label(gt, os.path.join(experiment_log_dir, "depth_gt.png"))
        
        writer.add_image('depth/pred', prediction, steps)
        writer.add_image('depth/gt', gt, steps)
        
    def accumulate_metric(self, prediction, gt, accumulator, distributed=False):
        if distributed:
            prediction_list = [torch.zeros_like(prediction).cuda() for _ in range(dist.get_world_size())]
            gt_list = [torch.zeros_like(gt).cuda() for _ in range(dist.get_world_size())]
            dist.all_gather(prediction_list, prediction)
            dist.all_gather(gt_list, gt)

            if local_rank == 0:
                prediction = torch.cat(prediction_list, dim=0)
                gt = torch.cat(gt_list, dim=0)

        se, depth_valid_pixels = squared_error(prediction, gt, 255)
        accumulator['squared_error'] = accumulator.get('squared_error', 0.) + se.item()
        accumulator['depth_valid_pixels'] = accumulator.get('depth_valid_pixels', 0.) + depth_valid_pixels
        return accumulator
    
    def aggregate_metric(self, accumulator):
        mse = accumulator['squared_error'] / accumulator['depth_valid_pixels']
        rmse = np.sqrt(mse)
        return {
            'RMSElog': rmse
        }
    
        
class ObjectClassTask(Task):
    
    def __init__(self, cfg):
        self.type = 'image'
        self.cfg = cfg
        self.name_list = np.load('datasets/taskonomy/selected_cls_names.npy')
        
    def loss(self, prediction, gt):
        # return class_loss(prediction, gt)
        return logits_loss(prediction, gt, True)
    
    def log_visualize(self, prediction, gt, loss, writer, steps):
        writer.add_scalar('loss/object', loss.data.item(), steps)
        value, idx = prediction.topk(5, 1, True, True)
        result = ""
        for v, i in zip(value[0], idx[0]):
            result += "%.4f_%s\n"%(v.cpu(), self.name_list[i.cpu()])
        writer.add_text('object/pred', result, steps)
        
    def accumulate_metric(self, prediction, gt, accumulator, distributed=False):
        if distributed:
            prediction_list = [torch.zeros_like(prediction).cuda() for _ in range(dist.get_world_size())]
            gt_list = [torch.zeros_like(gt).cuda() for _ in range(dist.get_world_size())]
            dist.all_gather(prediction_list, prediction)
            dist.all_gather(gt_list, gt)

            if local_rank == 0:
                prediction = torch.cat(prediction_list, dim=0)
                gt = torch.cat(gt_list, dim=0)
                
        mse = logits_loss(prediction, gt, False).mean(1).sum()
        _, cls_target = gt.max(1)
        top1_correct, top5_correct = count_correct(prediction, cls_target, topk=(1,5))
        accumulator['mse'] = accumulator.get('mse', 0.) + mse
        accumulator['total_object'] = accumulator.get('total_object', 0.) + gt.size(0)
        accumulator['correct_object_top1'] = accumulator.get('correct_object_top1', 0.) + top1_correct.item()
        accumulator['correct_object_top5'] = accumulator.get('correct_object_top5', 0.) + top5_correct.item()
        return accumulator
        
    def aggregate_metric(self, accumulator):
        mse = accumulator['mse'] / accumulator['total_object']
        accuracy_top1 = accumulator['correct_object_top1'] / accumulator['total_object']
        accuracy_top5 = accumulator['correct_object_top5'] / accumulator['total_object']
        return {
            'Object Logit MSE': mse,
            'Object Acc Top1': accuracy_top1,
            'Object Acc Top5': accuracy_top5
        }
        

class SceneClassTask(Task):
    
    def __init__(self, cfg):
        self.type = 'image'
        self.cfg = cfg
        self.name_list = np.load('datasets/taskonomy/selected_scene_names.npy')
        
    def loss(self, prediction, gt):
        # return class_loss(prediction, gt)
        return logits_loss(prediction, gt, True)
    
    def log_visualize(self, prediction, gt, loss, writer, steps):
        writer.add_scalar('loss/scene', loss.data.item(), steps)
        value, idx = prediction.topk(5, 1, True, True)
        result = ""
        for v, i in zip(value[0], idx[0]):
            result += "%.4f_%s\n"%(v.cpu(), self.name_list[i.cpu()])
        writer.add_text('scene/pred', result, steps)
        
    def accumulate_metric(self, prediction, gt, accumulator, distributed=False):
        if distributed:
            prediction_list = [torch.zeros_like(prediction).cuda() for _ in range(dist.get_world_size())]
            gt_list = [torch.zeros_like(gt).cuda() for _ in range(dist.get_world_size())]
            dist.all_gather(prediction_list, prediction)
            dist.all_gather(gt_list, gt)

            if local_rank == 0:
                prediction = torch.cat(prediction_list, dim=0)
                gt = torch.cat(gt_list, dim=0)
                
        mse = logits_loss(prediction, gt, False).mean(1).sum()
        _, cls_target = gt.max(1)
        top1_correct, top5_correct = count_correct(prediction, cls_target, topk=(1,5))
        accumulator['mse'] = accumulator.get('mse', 0.) + mse
        accumulator['total_scene'] = accumulator.get('total_scene', 0.) + gt.size(0)
        accumulator['correct_scene_top1'] = accumulator.get('correct_scene_top1', 0.) + top1_correct.item()
        accumulator['correct_scene_top5'] = accumulator.get('correct_scene_top5', 0.) + top5_correct.item()
        return accumulator
        
    def aggregate_metric(self, accumulator):
        mse = accumulator['mse'] / accumulator['total_scene']
        accuracy_top1 = accumulator['correct_scene_top1'] / accumulator['total_scene']
        accuracy_top5 = accumulator['correct_scene_top5'] / accumulator['total_scene']
        return {
            'Scene Logit MSE': mse,
            'Scene Acc Top1': accuracy_top1,
            'Scene Acc Top5': accuracy_top5
        }
        