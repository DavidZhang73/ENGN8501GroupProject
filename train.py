"""
Train
Adapted from https://github.com/PeterL1n/BackgroundMattingV2 by Jiahao Zhang
"""

import argparse
import os

import kornia
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from data_path import DATA_PATH
from dataset.augmentation import TrainFrameSpeedSampler, ValidFrameSampler
from dataset.video_matte import VideoMatte240KDataset, VideoMatteTrainAugmentation, VideoMatteValidAugmentation
from model import MattingBase
from model.utils import load_matched_state_dict

# --------------- Arguments ---------------


parser = argparse.ArgumentParser()

parser.add_argument('--dataset-name', type=str, default='videomatte8k', choices=DATA_PATH.keys())
parser.add_argument('--model-backbone', type=str, default='resnet50', choices=['resnet50'])
parser.add_argument('--model-name', type=str, default='convLSTM')
parser.add_argument('--model-pretrain-initialization', type=str, default=None)
parser.add_argument('--model-last-checkpoint', type=str, default=r'<path to last checkpoint>')

parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--seq-length', type=int, default=8)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--epoch-start', type=int, default=0)
parser.add_argument('--epoch-end', type=int, default=10)

parser.add_argument('--log-train-loss-interval', type=int, default=1)
parser.add_argument('--log-train-images-interval', type=int, default=20)
parser.add_argument('--log-valid-interval', type=int, default=1000)
parser.add_argument('--checkpoint-interval', type=int, default=1000)

args = parser.parse_args()


# --------------- Loading ---------------


def train():
    # Training DataLoader
    dataset_train = VideoMatte240KDataset(
        video_matte_path=DATA_PATH[args.dataset_name]['train'],
        background_image_path=DATA_PATH['backgrounds']['train'],
        seq_length=args.seq_length,
        seq_sampler=TrainFrameSpeedSampler(10),
        transform=VideoMatteTrainAugmentation((224, 224))
    )
    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Validation DataLoader
    dataset_valid = VideoMatte240KDataset(
        video_matte_path=DATA_PATH[args.dataset_name]['valid'],
        background_image_path=DATA_PATH['backgrounds']['valid'],
        seq_length=args.seq_length,
        seq_sampler=ValidFrameSampler(),
        transform=VideoMatteValidAugmentation((224, 224))
    )
    dataloader_valid = DataLoader(
        dataset_valid,
        pin_memory=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True
    )

    # Model
    model = MattingBase(args.model_backbone).cuda()

    if args.model_last_checkpoint is not None:
        load_matched_state_dict(model, torch.load(args.model_last_checkpoint))
    elif args.model_pretrain_initialization is not None:
        model.load_pretrained_deeplabv3_state_dict(torch.load(args.model_pretrain_initialization)['model_state'])

    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-4},
        {'params': model.aspp.parameters(), 'lr': 5e-4},
        {'params': model.decoder.parameters(), 'lr': 5e-4}
    ])
    scaler = GradScaler()

    # Logging and checkpoints
    if not os.path.exists(f'checkpoint/{args.model_name}'):
        os.makedirs(f'checkpoint/{args.model_name}')
    writer = SummaryWriter(f'log/{args.model_name}')

    # Run loop
    for epoch in range(args.epoch_start, args.epoch_end):
        bar = tqdm(dataloader_train, desc='[Train]')
        for i, (fgr, pha, bgr) in enumerate(bar):
            step = epoch * len(dataloader_train) + i + 1

            true_fgr = fgr.cuda(non_blocking=True)
            true_bgr = bgr.cuda(non_blocking=True)
            true_pha = pha.cuda(non_blocking=True)

            true_src = true_bgr.clone()

            # Composite foreground onto source
            true_src = true_fgr * true_pha + true_src * (1 - true_pha)

            with autocast():
                pred_pha, pred_fgr, pred_err = model(true_src)[:3]
                loss = compute_loss(
                    torch.flatten(pred_pha, 0, 1),
                    torch.flatten(pred_fgr, 0, 1),
                    torch.flatten(pred_err, 0, 1),
                    torch.flatten(true_pha, 0, 1),
                    torch.flatten(true_fgr, 0, 1)
                )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                bar.set_description(f'[Train] Epoch: {epoch}, Step: {step}, Loss: {loss:.4f}')

                if step == 1 or step % args.log_train_loss_interval == 0:
                    writer.add_scalar('loss', loss, step)

                if step == 1 or step % args.log_train_images_interval == 0:
                    writer.add_image(
                        'train_pred_pha',
                        make_grid(torch.flatten(pred_pha, 0, 1), nrow=args.seq_length),
                        step
                    )
                    writer.add_image(
                        'train_pred_fgr',
                        make_grid(torch.flatten(pred_fgr, 0, 1), nrow=args.seq_length),
                        step
                    )
                    writer.add_image(
                        'train_pred_err',
                        make_grid(torch.flatten(pred_err, 0, 1), nrow=args.seq_length),
                        step
                    )
                    writer.add_image(
                        'train_pred_com',
                        make_grid(torch.flatten(pred_fgr * pred_pha, 0, 1), nrow=args.seq_length),
                        step
                    )
                    writer.add_image(
                        'train_true_src',
                        make_grid(torch.flatten(true_src, 0, 1), nrow=args.seq_length),
                        step
                    )
                    writer.add_image(
                        'train_true_pha',
                        make_grid(torch.flatten(true_pha, 0, 1), nrow=args.seq_length),
                        step
                    )

                del true_pha, true_fgr, true_bgr
                del pred_pha, pred_fgr, pred_err

                if step % args.log_valid_interval == 0:
                    valid(model, dataloader_valid, writer, step)

                if step % args.checkpoint_interval == 0:
                    torch.save(model.state_dict(), f'checkpoint/{args.model_name}/epoch-{epoch}-iter-{step}.pth')

            torch.save(model.state_dict(), f'checkpoint/{args.model_name}/epoch-{epoch}.pth')

            # --------------- Utils ---------------


def compute_loss(pred_pha, pred_fgr, pred_err, true_pha, true_fgr):
    true_err = torch.abs(pred_pha.detach() - true_pha)
    true_msk = true_pha != 0
    return F.l1_loss(pred_pha, true_pha) + \
           F.l1_loss(kornia.sobel(pred_pha), kornia.sobel(true_pha)) + \
           F.l1_loss(pred_fgr * true_msk, true_fgr * true_msk) + \
           F.mse_loss(pred_err, true_err)


def valid(model, dataloader, writer, step):
    model.eval()
    loss_total = 0
    loss_count = 0
    with torch.no_grad():
        bar = tqdm(dataloader, desc='[Valid]')
        for true_fgr, true_pha, true_bgr in bar:
            batch_size = true_pha.size(0)

            true_pha = true_pha.cuda(non_blocking=True)
            true_fgr = true_fgr.cuda(non_blocking=True)
            true_bgr = true_bgr.cuda(non_blocking=True)
            true_src = true_pha * true_fgr + (1 - true_pha) * true_bgr

            pred_pha, pred_fgr, pred_err = model(true_src)[:3]
            loss = compute_loss(
                torch.flatten(pred_pha, 0, 1),
                torch.flatten(pred_fgr, 0, 1),
                torch.flatten(pred_err, 0, 1),
                torch.flatten(true_pha, 0, 1),
                torch.flatten(true_fgr, 0, 1)
            )
            bar.set_description(f'[Valid] Loss: {loss:.4f}')
            loss_total += loss.cpu().item() * batch_size
            loss_count += batch_size

    writer.add_scalar('valid_loss', loss_total / loss_count, step)
    model.train()


# --------------- Start ---------------


if __name__ == '__main__':
    train()
