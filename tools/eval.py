import os
import argparse
import random
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F

# ad-hoc way to deal with python 3.7.4
import os, sys
lib_path = os.path.abspath(os.path.join('.'))
print(lib_path)
sys.path.append(lib_path)

from core.config import cfg
from core.tasks import get_tasks
from core.data import get_dataset
from core.models import get_model


def evaluate(test_loader, model, task1, task2, distributed=False, local_rank=None):
    if distributed:
        assert local_rank is not None

    with torch.no_grad():
        accumulator1 = {}
        accumulator2 = {}
        for batch_idx, (image, label_1, label_2) in enumerate(test_loader):
            if cfg.CUDA:
                image, label_1, label_2 = image.cuda(), label_1.cuda(), label_2.cuda()

            result = model(image)
            out1, out2 = result.out1, result.out2
                
            accumulator1 = task1.accumulate_metric(out1, label_1, accumulator1, distributed)
            accumulator2 = task2.accumulate_metric(out2, label_2, accumulator2, distributed)

        task1_metric = task1.aggregate_metric(accumulator1)
        task2_metric = task2.aggregate_metric(accumulator2)

        return task1_metric, task2_metric


def main():
    parser = argparse.ArgumentParser(description="Baseline Experiment Eval")
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
    
    # Seeding
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # This can slow down training

    # load the data
    test_loader = torch.utils.data.DataLoader(
        get_dataset(cfg, 'test'),
        batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, pin_memory=True)

    task1, task2 = get_tasks(cfg)
    model = get_model(cfg, task1, task2)
    
    ckpt_path = os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME, 'ckpt-%s.pth' % str(cfg.TEST.CKPT_ID).zfill(5))
    print("Evaluating Checkpoint at %s" % ckpt_path)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    if cfg.CUDA:
        model = model.cuda()

    task1_metric, task2_metric = evaluate(test_loader, model, task1, task2)
    for k, v in task1_metric.items():
        print('{}: {:.3f}'.format(k, v))
    for k, v in task2_metric.items():
        print('{}: {:.3f}'.format(k, v))


if __name__ == '__main__':
    main()
