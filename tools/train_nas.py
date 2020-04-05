import os
import random
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from tensorboardX import SummaryWriter

# from apex import amp

# ad-hoc way to deal with python 3.7.4
import os, sys
lib_path = os.path.abspath(os.path.join('.'))
print(lib_path)
sys.path.append(lib_path)

from core.config import cfg
from core.data import get_dataset
from core.tasks import get_tasks
from core.models import get_model
from core.utils import get_print
from core.utils.losses import entropy_loss, l1_loss
from core.utils.visualization import process_image, save_heatmap, save_connectivity

from eval import evaluate


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


class MyDataParallel(nn.parallel.DistributedDataParallel):
    """
    Wrapper to grant access to class attributes
    """

    def __init__(self, *arg, **kwarg):
        super(MyDataParallel, self).__init__(*arg, **kwarg)

    def __getattr__(self, name):
        if name == 'module':
            return nn.parallel.DistributedDataParallel.__getattr__(self, name)
        return getattr(self.module, name)


def main():
    parser = argparse.ArgumentParser(description="PyTorch MTLNAS Training")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=29501)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # Preparing for DDP training
    logging = args.local_rank == 0
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(args.port)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.EXPERIMENT_NAME = args.config_file.split('/')[-1][:-5]
    cfg.merge_from_list(args.opts)
    # Adjust batch size for distributed training
    assert cfg.TRAIN.BATCH_SIZE % num_gpus == 0
    cfg.TRAIN.BATCH_SIZE = int(cfg.TRAIN.BATCH_SIZE // num_gpus)
    assert cfg.TEST.BATCH_SIZE % num_gpus == 0
    cfg.TEST.BATCH_SIZE = int(cfg.TEST.BATCH_SIZE // num_gpus)
    cfg.freeze()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d~%H:%M:%S")
    experiment_log_dir = os.path.join(cfg.LOG_DIR, cfg.EXPERIMENT_NAME, timestamp)
    if not os.path.exists(experiment_log_dir) and logging:
        os.makedirs(experiment_log_dir)
        writer = SummaryWriter(logdir=experiment_log_dir)
    printf = get_print(experiment_log_dir)
    printf("Training with Config: ")
    printf(cfg)

    # Seeding
    os.environ['PYTHONHASHSEED'] = str(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # This can slow down training

    if not os.path.exists(os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME)) and logging:
        os.makedirs(os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME))

    # load the data
    train_full_data = get_dataset(cfg, 'train')

    num_train = len(train_full_data)
    indices = list(range(num_train))
    split = int(np.floor(cfg.ARCH.TRAIN_SPLIT * num_train))

    # load the data
    if cfg.TRAIN.EVAL_CKPT:
        test_data = get_dataset(cfg, 'val')

        if distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
        else:
            test_sampler = None

        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, sampler=test_sampler)

    task1, task2 = get_tasks(cfg)
    model = get_model(cfg, task1, task2)

    if cfg.CUDA:
        model = model.cuda()

    if distributed:
        # Important: Double check if BN is working as expected
        if cfg.TRAIN.APEX:
            printf("using apex synced BN")
            model = apex.parallel.convert_syncbn_model(model)
        else:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = MyDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # hacky way to pick params
    nddr_params = []
    fc8_weights = []
    fc8_bias = []
    base_params = []
    for k, v in model.named_net_parameters():
        if 'paths' in k:
            nddr_params.append(v)
        elif model.net1.fc_id in k:
            if 'weight' in k:
                fc8_weights.append(v)
            else:
                assert 'bias' in k
                fc8_bias.append(v)
        else:
            assert 'alpha' not in k
            base_params.append(v)
    assert len(nddr_params) > 0 and len(fc8_weights) > 0 and len(fc8_bias) > 0

    parameter_dict = [
        {'params': base_params},
        {'params': fc8_weights, 'lr': cfg.TRAIN.LR * cfg.TRAIN.FC8_WEIGHT_FACTOR},
        {'params': fc8_bias, 'lr': cfg.TRAIN.LR * cfg.TRAIN.FC8_BIAS_FACTOR},
        {'params': nddr_params, 'lr': cfg.TRAIN.LR * cfg.TRAIN.NDDR_FACTOR}
    ]
    optimizer = optim.SGD(parameter_dict, lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    
    if cfg.ARCH.OPTIMIZER == 'sgd':
        arch_optimizer = torch.optim.SGD(model.arch_parameters(),
                                         lr=cfg.ARCH.LR,
                                         momentum=cfg.TRAIN.MOMENTUM,  # TODO: separate this param
                                         weight_decay=cfg.ARCH.WEIGHT_DECAY)
    else:
        arch_optimizer = torch.optim.Adam(model.arch_parameters(),
                                          lr=cfg.ARCH.LR,
                                          betas=(0.5, 0.999),
                                          weight_decay=cfg.ARCH.WEIGHT_DECAY)

    if cfg.TRAIN.SCHEDULE == 'Poly':
        if cfg.TRAIN.WARMUP > 0.:
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lambda step: min(1., float(step) / cfg.TRAIN.WARMUP) * (1 - float(step) / cfg.TRAIN.STEPS) ** cfg.TRAIN.POWER,
                                                    last_epoch=-1)
            arch_scheduler = optim.lr_scheduler.LambdaLR(arch_optimizer,
                                                    lambda step: min(1., float(step) / cfg.TRAIN.WARMUP) * (1 - float(step) / cfg.TRAIN.STEPS) ** cfg.TRAIN.POWER,
                                                    last_epoch=-1)
        else:
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lambda step: (1 - float(step) / cfg.TRAIN.STEPS) ** cfg.TRAIN.POWER,
                                                    last_epoch=-1)
            arch_scheduler = optim.lr_scheduler.LambdaLR(arch_optimizer,
                                                    lambda step: (1 - float(step) / cfg.TRAIN.STEPS) ** cfg.TRAIN.POWER,
                                                    last_epoch=-1)
    elif cfg.TRAIN.SCHEDULE == 'Cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.STEPS)
        arch_scheduler = optim.lr_scheduler.CosineAnnealingLR(arch_optimizer, cfg.TRAIN.STEPS)
    elif cfg.TRAIN.SCHEDULE == 'Step':
        milestones = (np.array([0.6, 0.9]) * cfg.TRAIN.STEPS).astype('int')
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
        arch_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    else:
        raise NotImplementedError
        
    if cfg.TRAIN.APEX:
        model, [arch_optimizer, optimizer] = amp.initialize(model, [arch_optimizer, optimizer], opt_level="O1", num_losses=2)

    model.train()
    steps = 0
    while steps < cfg.TRAIN.STEPS:
        # Initialize train/val dataloader below this shuffle operation
        # to ensure both arch and weights gets to see all the data,
        # but not at the same time during mixed data training
        if cfg.ARCH.MIXED_DATA:
            np.random.shuffle(indices)

        train_data = torch.utils.data.Subset(train_full_data, indices[:split])
        val_data = torch.utils.data.Subset(train_full_data, indices[split:num_train])

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=cfg.TRAIN.BATCH_SIZE,
            pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=cfg.TRAIN.BATCH_SIZE,
            pin_memory=True, sampler=val_sampler)

        val_iter = iter(val_loader)

        if distributed:
            train_sampler.set_epoch(steps)  # steps is used to seed RNG
            val_sampler.set_epoch(steps)

        for batch_idx, (image, label_1, label_2) in enumerate(train_loader):
            if cfg.CUDA:
                image, label_1, label_2 = image.cuda(), label_1.cuda(), label_2.cuda()

            # get a random minibatch from the search queue without replacement
            val_batch = next(val_iter, None)
            if val_batch is None:  # val_iter has reached its end
                val_sampler.set_epoch(steps)
                val_iter = iter(val_loader)
                val_batch = next(val_iter)
            image_search, label_1_search, label_2_search = val_batch
            image_search = image_search.cuda()
            label_1_search, label_2_search = label_1_search.cuda(), label_2_search.cuda()

            # setting flag for training arch parameters
            model.arch_train()
            assert model.arch_training
            arch_optimizer.zero_grad()
            arch_result = model.loss(image_search, (label_1_search, label_2_search))
            arch_loss = arch_result.loss
            
            
            # Mixed Precision
            if cfg.TRAIN.APEX:
                with amp.scale_loss(arch_loss, arch_optimizer, loss_id=0) as scaled_loss:
                    scaled_loss.backward()
            else:
                arch_loss.backward()
                
            arch_optimizer.step()
            model.arch_eval()

            assert not model.arch_training
            optimizer.zero_grad()
            
            result = model.loss(image, (label_1, label_2))

            out1, out2 = result.out1, result.out2
            loss1 = result.loss1
            loss2 = result.loss2

            loss = result.loss
            
            # Mixed Precision
            if cfg.TRAIN.APEX:
                with amp.scale_loss(loss, optimizer, loss_id=1) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            if cfg.ARCH.SEARCHSPACE == 'GeneralizedMTLNAS':
                model.step()  # update model temperature
            scheduler.step()
            if cfg.ARCH.OPTIMIZER == 'sgd':
                arch_scheduler.step()

            # Print out the loss periodically.
            if steps % cfg.TRAIN.LOG_INTERVAL == 0 and logging:
                printf('Train Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                    steps, batch_idx * len(image), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item(),
                    loss1.data.item(), loss2.data.item()))

                # Log to tensorboard
                writer.add_scalar('lr', scheduler.get_lr()[0], steps)
                writer.add_scalar('arch_lr', arch_scheduler.get_lr()[0], steps)
                writer.add_scalar('loss/overall', loss.data.item(), steps)
                writer.add_image('image', process_image(image[0], train_full_data.image_mean), steps)
                task1.log_visualize(out1, label_1, loss1, writer, steps)
                task2.log_visualize(out2, label_2, loss2, writer, steps)
                
                if cfg.ARCH.ENTROPY_REGULARIZATION:
                    writer.add_scalar('loss/entropy_weight', arch_result.entropy_weight, steps)
                    writer.add_scalar('loss/entropy_loss', arch_result.entropy_loss.data.item(), steps)

                if cfg.ARCH.L1_REGULARIZATION:
                    writer.add_scalar('loss/l1_weight', arch_result.l1_weight, steps)
                    writer.add_scalar('loss/l1_loss', arch_result.l1_loss.data.item(), steps)

                if cfg.ARCH.SEARCHSPACE == 'GeneralizedMTLNAS':
                    writer.add_scalar('temperature', model.get_temperature(), steps)
                    alpha1 = torch.sigmoid(model.net1_alphas).detach().cpu().numpy()
                    alpha2 = torch.sigmoid(model.net2_alphas).detach().cpu().numpy()
                    alpha1_path = os.path.join(experiment_log_dir, 'alpha1')
                    if not os.path.isdir(alpha1_path):
                        os.makedirs(alpha1_path)
                    alpha2_path = os.path.join(experiment_log_dir, 'alpha2')
                    if not os.path.isdir(alpha2_path):
                        os.makedirs(alpha2_path)
                    heatmap1 = save_heatmap(alpha1, os.path.join(alpha1_path, "%s_alpha1.png"%str(steps).zfill(5)))
                    heatmap2 = save_heatmap(alpha2, os.path.join(alpha2_path, "%s_alpha2.png"%str(steps).zfill(5)))
                    writer.add_image('alpha/net1', heatmap1, steps)
                    writer.add_image('alpha/net2', heatmap2, steps)
                    network_path = os.path.join(experiment_log_dir, 'network')
                    if not os.path.isdir(network_path):
                        os.makedirs(network_path)
                    connectivity_plot = save_connectivity(alpha1, alpha2,
                                                          model.net1_connectivity_matrix,
                                                          model.net2_connectivity_matrix,
                                                          os.path.join(network_path, "%s_network.png"%str(steps).zfill(5))
                                                         )
                    writer.add_image('network', connectivity_plot, steps)
                

            if steps % cfg.TRAIN.EVAL_INTERVAL == 0:
                if distributed:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()

                checkpoint = {
                    'cfg': cfg,
                    'step': steps,
                    'model_state_dict': state_dict,
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
                    torch.cuda.empty_cache()  # TODO check if it helps
                    task1_metric, task2_metric = evaluate(test_loader, model, task1, task2, distributed, args.local_rank)

                    if logging:
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
                    torch.cuda.empty_cache()  # TODO check if it helps

                if logging and steps % cfg.TRAIN.SAVE_INTERVAL == 0:
                    torch.save(checkpoint, os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME,
                                                        'ckpt-%s.pth' % str(steps).zfill(5)))

            if steps >= cfg.TRAIN.STEPS:
                break
            steps += 1  # train for one extra iteration to allow time for tensorboard logging..


if __name__ == '__main__':
    main()
