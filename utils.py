import argparse
import os
import os.path as osp
import random
from enum import Enum
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop


def parse_args():
    parser = argparse.ArgumentParser(description="BrightFormer")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/LOL",
        help="Path to dataset",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="result",
        help="path to save checkpoint and log",
    )
    parser.add_argument("--num_epoch", type=int, default=300)
    parser.add_argument(
        "--batch_size",
        nargs="+",
        type=int,
        default=[16, 10, 8, 4, 2, 2],
        help="batch size of loading images for progressive learning",
    )
    parser.add_argument(
        "--patch_size",
        nargs="+",
        type=int,
        default=[128, 160, 192, 256, 320, 384],
        help="patch size of each image for progressive learning",
    )
    parser.add_argument(
        "--milestone",
        nargs="+",
        type=int,
        default=[90, 156, 204, 240, 276],
        help="when to change patch size and batch size",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=2e-2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--seed", type=int, default=-1, help="random seed (-1 for no manual seed)"
    )
    parser.add_argument("--pad_multiple_to", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--eval_step", type=int, default=20)
    parser.add_argument("--save_step", type=int, default=20)
    parser.add_argument("--phase", choices=["train", "test"], default="train")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")

    return init_args(parser.parse_args())


def init_args(args):
    if args.phase == "test" and args.ckpt is None:
        assert "checkpoint should be specified in the test phase"

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(osp.join(args.save_path, "checkpoints"), exist_ok=True)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    return args


class LOLDataset(Dataset):
    def __init__(
        self,
        data_path,
        input_dir="low",
        gt_dir="high",
        training=True,
        patch_size=128,
        size_divisibility=8,
        enlarge_factor=1,
    ):
        super(LOLDataset, self).__init__()

        self.dataroot = data_path
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.fns = os.listdir(osp.join(data_path, gt_dir))
        self.training = training
        self.patch_size = patch_size
        self.size_divisibility = size_divisibility
        self.enlarge_factor = enlarge_factor

    def __getitem__(self, i):
        fn = self.fns[i % len(self.fns)]
        input_path = osp.join(self.dataroot, self.input_dir, fn)
        target_path = osp.join(self.dataroot, self.gt_dir, fn)

        input = T.to_tensor(Image.open(input_path))
        target = T.to_tensor(Image.open(target_path))

        if self.training:
            i, j, th, tw = RandomCrop.get_params(
                input, (self.patch_size, self.patch_size)
            )
            input = T.crop(input, i, j, th, tw)
            target = T.crop(target, i, j, th, tw)

            augments = [
                None,
                T.hflip,
                T.vflip,
                partial(T.rotate, angle=90),
                partial(T.rotate, angle=180),
                partial(T.rotate, angle=270),
            ]
            aug = random.choice(augments)
            if aug:
                input = aug(input)
                target = aug(target)
        else:
            h, w = input.shape[-2:]
            new_h = (
                (h + self.size_divisibility) // self.size_divisibility
            ) * self.size_divisibility
            new_w = (
                (w + self.size_divisibility) // self.size_divisibility
            ) * self.size_divisibility
            pad_h = new_h - h if h % self.size_divisibility != 0 else 0
            pad_w = new_w - w if w % self.size_divisibility != 0 else 0
            input = F.pad(input, (0, pad_w, 0, pad_h), "reflect")

        return {"input": input, "target": target, "fn": osp.splitext(fn)[0]}

    def __len__(self):
        return len(self.fns) * self.enlarge_factor


class AverageMeter(object):
    """Computes and stores the average and current value"""

    class Summary(Enum):
        NONE = 0
        AVERAGE = 1
        SUM = 2
        COUNT = 3

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is self.Summary.NONE:
            fmtstr = ""
        elif self.summary_type is self.Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is self.Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is self.Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError(f"invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)
