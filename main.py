import os.path as osp
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import BrightFormer
from utils import AverageMeter, LOLDataset, parse_args


def evaluate_model(net, data_loader, save_dir, num_epoch=None):
    net.eval()
    avg_psnr, avg_ssim = AverageMeter('PSNR'), AverageMeter('SSIM')

    with torch.inference_mode():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for data in test_bar:
            lq, gt, fn = (
                data['input'].to(device),
                data['target'].to(device),
                data['fn'][0],
            )

            out = torch.clamp(net(lq)[0], 0.0, 1.0)
            out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt = gt.squeeze(0).permute(1, 2, 0).cpu().numpy()

            out = (out * 255).astype(np.uint8)
            gt = (gt * 255).astype(np.uint8)

            h, w, _ = gt.shape
            out = out[:h, :w, :]

            current_psnr = compute_psnr(out, gt)
            current_ssim = compute_ssim(
                out,
                gt,
                channel_axis=2,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )

            avg_psnr.update(current_psnr)
            avg_ssim.update(current_ssim)

            save_path = Path(save_dir, 'val_result', str(num_epoch), f'{fn}.png')
            save_path.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray(out).save(save_path)
            test_bar.set_description(
                f'Test Epoch: [{num_epoch}] '
                f'PSNR: {avg_psnr.avg:.2f} SSIM: {avg_ssim.avg:.4f}'
            )

    return avg_psnr.avg, avg_ssim.avg


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cpu') if args.cpu else torch.device('cuda')

    test_dataset = LOLDataset(
        osp.join(args.data_path, 'eval15'),
        training=False,
        size_divisibility=args.pad_multiple_to,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    save_path = Path(args.save_path)

    model = BrightFormer().to(device)
    if args.phase == 'test':
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        evaluate_model(model, test_loader, save_path, 'final')
    else:
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.num_epoch, eta_min=args.min_lr
        )
        total_loss, total_num, i = 0.0, 0, 0
        for n_epoch in range(1, args.num_epoch + 1):
            # progressive learning
            if n_epoch == 1 or n_epoch - 1 in args.milestone:
                end_epoch = args.milestone[i] if i < len(args.milestone) else args.num_epoch
                start_epoch = args.milestone[i - 1] if i > 0 else 0
                length = args.batch_size[i] * (end_epoch - start_epoch)
                train_dataset = LOLDataset(osp.join(args.data_path, 'our485'),training=True, patch_size=args.patch_size[i])
                train_loader = DataLoader(train_dataset, args.batch_size[i], shuffle=True, num_workers=args.workers)
                i += 1

            train_bar = tqdm(train_loader, dynamic_ncols=True)
            for data in train_bar:  # train
                model.train()
                lq, gt = data['input'].to(device), data['target'].to(device)
                gt_illu = ((gt - lq) / (gt + 1e-5))
                out, illu = model(lq)
                loss = F.l1_loss(out, gt) + F.l1_loss(illu, gt_illu)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
                optimizer.step()

                total_num += lq.size(0)
                total_loss += loss.item() * lq.size(0)
                train_bar.set_description(
                    f'Train Epoch: [{n_epoch}/{args.num_epoch+1}] Loss: {total_loss / total_num:.3f}'
                )

            lr_scheduler.step()

            if n_epoch % args.eval_step == 0:  # evaluate
                val_psnr, val_ssim = evaluate_model(
                    model, test_loader, save_path, n_epoch
                )

                # save statistics
                with save_path.joinpath('record.txt').open(mode='a+') as f:
                    f.write(
                        f'Epoch: {n_epoch} PSNR:{val_psnr:.2f} SSIM:{val_ssim:.4f}\n'
                    )

            if n_epoch % args.save_step == 0:
                torch.save(
                    model.state_dict(),
                    save_path.joinpath('checkpoints', f'{n_epoch}.pth'),
                )
