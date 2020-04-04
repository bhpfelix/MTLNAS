import os
import random
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

# ad-hoc way to deal with python 3.7.4
import os, sys
lib_path = os.path.abspath(os.path.join('.'))
print(lib_path)
sys.path.append(lib_path)

from core.config import cfg
from core.tasks import get_tasks
from core.data import get_dataset
from core.models import get_model
from core.utils import get_print
from core.utils.visualization import process_image

from eval import evaluate


def main():
    parser = argparse.ArgumentParser(description="Baseline Experiment Training")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.EXPERIMENT_NAME = args.config_file.split('/')[-1][:-5]
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d~%H:%M:%S")
    experiment_log_dir = os.path.join(cfg.LOG_DIR, cfg.EXPERIMENT_NAME, timestamp)
    if not os.path.exists(experiment_log_dir):
        os.makedirs(experiment_log_dir)
    writer = SummaryWriter(logdir=experiment_log_dir)
    printf = get_print(experiment_log_dir)
    printf("Training with Config: ")
    printf(cfg)

    # Seeding
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # This can slow down training

    if not os.path.exists(os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME)):
        os.makedirs(os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME))

    # load the data
    train_data = get_dataset(cfg, 'train')
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, pin_memory=True)

    # load the data
    if cfg.TRAIN.EVAL_CKPT:
        test_loader = torch.utils.data.DataLoader(
            get_dataset(cfg, 'val'),
            batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, pin_memory=True)

    task1, task2 = get_tasks(cfg)
    model = get_model(cfg, task1, task2)
    
    if cfg.CUDA:
        model = model.cuda()

    # hacky way to pick params
    nddr_params = []
    fc8_weights = []
    fc8_bias = []
    base_params = []
    for k, v in model.named_parameters():
        if 'nddrs' in k:
            nddr_params.append(v)
        elif model.net1.fc_id in k:
            if 'weight' in k:
                fc8_weights.append(v)
            else:
                assert 'bias' in k
                fc8_bias.append(v)
        else:
            base_params.append(v)
    
    if not cfg.MODEL.SINGLETASK and not cfg.MODEL.SHAREDFEATURE:
        assert len(nddr_params) > 0 and len(fc8_weights) > 0 and len(fc8_bias) > 0

    parameter_dict = [
        {'params': fc8_weights, 'lr': cfg.TRAIN.LR * cfg.TRAIN.FC8_WEIGHT_FACTOR},
        {'params': fc8_bias, 'lr': cfg.TRAIN.LR * cfg.TRAIN.FC8_BIAS_FACTOR},
        {'params': nddr_params, 'lr': cfg.TRAIN.LR * cfg.TRAIN.NDDR_FACTOR}
    ]
    
    if not cfg.TRAIN.FREEZE_BASE:
        parameter_dict.append({'params': base_params})
    else:
        printf("Frozen net weights")
        
    optimizer = optim.SGD(parameter_dict, lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    if cfg.TRAIN.SCHEDULE == 'Poly':
        if cfg.TRAIN.WARMUP > 0.:
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lambda step: min(1., float(step) / cfg.TRAIN.WARMUP) * (1 - float(step) / cfg.TRAIN.STEPS) ** cfg.TRAIN.POWER,
                                                    last_epoch=-1)
        else:
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lambda step: (1 - float(step) / cfg.TRAIN.STEPS) ** cfg.TRAIN.POWER,
                                                    last_epoch=-1)
    elif cfg.TRAIN.SCHEDULE == 'Cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.STEPS)
    else:
        raise NotImplementedError
        
    if cfg.TRAIN.APEX:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    model.train()
    steps = 0
    while steps < cfg.TRAIN.STEPS:
        for batch_idx, (image, label_1, label_2) in enumerate(train_loader):
            if cfg.CUDA:
                image, label_1, label_2 = image.cuda(), label_1.cuda(), label_2.cuda()
            optimizer.zero_grad()

            result = model.loss(image, (label_1, label_2))
            out1, out2 = result.out1, result.out2

            loss1 = result.loss1
            loss2 = result.loss2

            loss = result.loss
            
            # Mixed Precision
            if cfg.TRAIN.APEX:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            model.step()  # update model step count
            scheduler.step()

            # Print out the loss periodically.
            if steps % cfg.TRAIN.LOG_INTERVAL == 0:
                printf('Train Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                    steps, batch_idx * len(image), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item(),
                    loss1.data.item(), loss2.data.item()))

                # Log to tensorboard
                writer.add_scalar('lr', scheduler.get_lr()[0], steps)
                writer.add_scalar('loss/overall', loss.data.item(), steps)
                task1.log_visualize(out1, label_1, loss1, writer, steps)
                task2.log_visualize(out2, label_2, loss2, writer, steps)
                writer.add_image('image', process_image(image[0], train_data.image_mean), steps)

            if steps % cfg.TRAIN.SAVE_INTERVAL == 0:
                checkpoint = {
                    'cfg': cfg,
                    'step': steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                    'loss1': loss1,
                    'loss2': loss2,
                    'task1_metric': None,
                    'task2_metric': None,
                }

                if cfg.TRAIN.EVAL_CKPT:
                    model.eval()
                    task1_metric, task2_metric = evaluate(test_loader, model, task1, task2)
                    for k, v in task1_metric.items():
                        writer.add_scalar('eval/{}'.format(k), v, steps)
                    for k, v in task2_metric.items():
                        writer.add_scalar('eval/{}'.format(k), v, steps)
                    for k, v in task1_metric.items():
                        printf('{}: {:.3f}'.format(k, v))
                    for k, v in task2_metric.items():
                        printf('{}: {:.3f}'.format(k, v))

                    checkpoint['task1_metric'] = task1_metric
                    checkpoint['task2_metric'] = task2_metric
                    model.train()

                torch.save(checkpoint, os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME,
                                                    'ckpt-%s.pth' % str(steps).zfill(5)))

            if steps >= cfg.TRAIN.STEPS:
                break
            steps += 1


if __name__ == '__main__':
    main()
