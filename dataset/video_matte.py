"""
Dataset
Adapted from https://github.com/PeterL1n/RobustVideoMatting by Jiahao Zhang
"""
import os
import random

from PIL import Image
from torch.utils.data import Dataset

from dataset.augmentation import MotionAugmentation


class VideoMatte240KDataset(Dataset):
    def __init__(self,
                 video_matte_path,
                 background_image_path,
                 seq_length,
                 seq_sampler,
                 transform=None,
                 background_image_id=None):
        self.background_image_path = background_image_path
        self.background_image_files = os.listdir(background_image_path)
        self.video_matte_path = video_matte_path
        self.video_matte_clips = sorted(os.listdir(os.path.join(video_matte_path, 'fgr')))
        self.video_matte_frames = [sorted(os.listdir(os.path.join(video_matte_path, 'fgr', clip)))
                                   for clip in self.video_matte_clips]
        self.video_matte_idx = [(clip_idx, frame_idx)
                                for clip_idx in range(len(self.video_matte_clips))
                                for frame_idx in range(0, len(self.video_matte_frames[clip_idx]), seq_length)]
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform
        self.background_image_id = background_image_id

    def __len__(self):
        return len(self.video_matte_idx)

    def __getitem__(self, idx):
        if self.background_image_id is not None:
            bgrs = self._get_image_background()
        else:
            bgrs = self._get_random_image_background()
        fgrs, phas = self._get_video_matte(idx)
        if self.transform is not None:
            return self.transform(fgrs, phas, bgrs)
        return fgrs, phas, bgrs

    def _get_image_background(self):
        with Image.open(
                os.path.join(
                    self.background_image_path,
                    self.background_image_files[self.background_image_id]
                )
        ) as bgr:
            bgr = bgr.convert('RGB')
        bgrs = [bgr] * self.seq_length
        return bgrs

    def _get_random_image_background(self):
        with Image.open(os.path.join(self.background_image_path, random.choice(self.background_image_files))) as bgr:
            bgr = bgr.convert('RGB')
        bgrs = [bgr] * self.seq_length
        return bgrs

    def _get_video_matte(self, idx):
        clip_idx, frame_idx = self.video_matte_idx[idx]
        clip = self.video_matte_clips[clip_idx]
        frame_count = len(self.video_matte_frames[clip_idx])
        fgrs, phas = [], []
        for i in self.seq_sampler(self.seq_length):
            frame = self.video_matte_frames[clip_idx][(frame_idx + i) % frame_count]
            with Image.open(os.path.join(self.video_matte_path, 'fgr', clip, frame)) as fgr, \
                    Image.open(os.path.join(self.video_matte_path, 'pha', clip, frame)) as pha:
                fgr = fgr.convert('RGB')
                pha = pha.convert('L')
            fgrs.append(fgr)
            phas.append(pha)
        return fgrs, phas


class VideoMatteTrainAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0.3,
            prob_bgr_affine=0.3,
            prob_noise=0.1,
            prob_color_jitter=0.3,
            prob_grayscale=0.02,
            prob_sharpness=0.1,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
        )


class VideoMatteValidAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0,
            prob_bgr_affine=0,
            prob_noise=0,
            prob_color_jitter=0,
            prob_grayscale=0,
            prob_sharpness=0,
            prob_blur=0,
            prob_hflip=0,
            prob_pause=0,
        )
