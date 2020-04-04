import os
import argparse
import random
import numpy as np
import random

import torch
import torch.nn as nn
import torch.distributed as dist

# ad-hoc way to deal with python 3.7.4
import os, sys
lib_path = os.path.abspath(os.path.join('.'))
print(lib_path)
sys.path.append(lib_path)

from core.config import cfg
from core.tasks import get_tasks
from core.data import get_dataset
from core.models import get_model

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
    parser = argparse.ArgumentParser(description="PyTorch MTLNAS Eval")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=29502)
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

    # Seeding
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
    test_data = get_dataset(cfg, 'test')

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

    ckpt_path = os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME, 'ckpt-%s.pth' % str(cfg.TEST.CKPT_ID).zfill(5))
    print("Evaluating Checkpoint at %s" % ckpt_path)
    ckpt = torch.load(ckpt_path)
    # compatibility with ddp saved checkpoint when evaluating without ddp
    pretrain_dict = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model_dict = model.state_dict()
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = MyDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    model.eval()

    task1_metric, task2_metric = evaluate(test_loader, model, task1, task2, distributed, args.local_rank)
    if logging:
        for k, v in task1_metric.items():
            print('{}: {:.9f}'.format(k, v))
        for k, v in task2_metric.items():
            print('{}: {:.9f}'.format(k, v))


if __name__ == '__main__':
    main()
