"""
Validation
Implemented by Peng Zhang
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import utils as v_utils
from tqdm import tqdm

from data_path import DATA_PATH
from dataset.augmentation import ValidFrameSampler, ValidAugmentation
from dataset.video_matte import VideoMatte240KDataset
from model import MattingBase
from model.utils import load_matched_state_dict

# --------------- Arguments ---------------


parser = argparse.ArgumentParser()

parser.add_argument('--dataset-name', type=str, default='videomatte8k', choices=DATA_PATH.keys())
parser.add_argument('--model-backbone', type=str, default='resnet50', choices=['resnet50'])
parser.add_argument('--model-checkpoint', type=str, default=r'<path to checkpoint>')
parser.add_argument('--output-path', type=str, default=r'<path to output>')
parser.add_argument('--seq-length', type=int, default=1)
parser.add_argument('--num-workers', type=int, default=0)

args = parser.parse_args()

# --------------- Loading ---------------

dataset_valid = VideoMatte240KDataset(
    video_matte_path=DATA_PATH[args.dataset_name]['valid'],
    background_image_path=DATA_PATH['backgrounds']['valid'],
    seq_length=args.seq_length,
    seq_sampler=ValidFrameSampler(),
    transform=ValidAugmentation((224, 224)),
    background_image_id=142
)
dataloader_valid = DataLoader(
    dataset_valid,
    pin_memory=False,
    batch_size=1,
    num_workers=args.num_workers
)

# Model
model = MattingBase(args.model_backbone).cuda()
load_matched_state_dict(model, torch.load(args.model_checkpoint))
model.eval()


# Validate
def save_img_tensor_list(t, start_index, output_dir):
    output_path = os.path.join(args.output_path, output_dir)
    os.makedirs(output_path, exist_ok=True)
    index = start_index
    for img in t[0]:
        v_utils.save_image(img, os.path.join(output_path, f'{index:06d}.png'))
        index += 1


os.makedirs(args.output_path, exist_ok=True)
with torch.no_grad():
    for i, (fgr, pha, bgr) in enumerate(tqdm(dataset_valid)):
        true_fgr = fgr.unsqueeze(0).cuda(non_blocking=True)
        true_bgr = bgr.unsqueeze(0).cuda(non_blocking=True)
        true_pha = pha.unsqueeze(0).cuda(non_blocking=True)
        true_src = true_bgr.clone()
        true_src = true_fgr * true_pha + true_src * (1 - true_pha)
        pred_pha, pred_fgr, pred_err = model(true_src)[:3]

        state = model.decoder.state3

        index_start = i * args.seq_length
        for i in range(state[0][0].size()[1]):
            save_img_tensor_list(state[0][0][:, i, :, :].unsqueeze(0), index_start, f'state_h_{i}')
            save_img_tensor_list(state[0][1][:, i, :, :].unsqueeze(0), index_start, f'state_c_{i}')
        save_img_tensor_list(pred_pha, index_start, 'pred_pha')
        save_img_tensor_list(true_pha, index_start, 'true_pha')
        save_img_tensor_list(pred_err, index_start, 'pred_err')
        save_img_tensor_list(true_src, index_start, 'true_src')
