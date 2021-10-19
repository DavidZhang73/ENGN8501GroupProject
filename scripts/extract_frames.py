import os
from os import path

from tqdm.contrib.concurrent import thread_map

VIDEO_PATH = r'D:\Downloads\Background Matting\dataset\VideoMatte240KVideo'
FRAME_PATH = r'D:\Downloads\Background Matting\dataset\VideoMatte240KFrames'


def get_task_list():
    ret = []
    for mode in ['train', 'test']:
        for fgr_name in os.listdir(path.join(VIDEO_PATH, mode, 'fgr')):
            name, _ = fgr_name.split('.')
            ret.append((
                path.join(VIDEO_PATH, mode, 'fgr', fgr_name),
                path.join(VIDEO_PATH, mode, 'pha', fgr_name),
                path.join(FRAME_PATH, mode, 'fgr', name),
                path.join(FRAME_PATH, mode, 'pha', name)
            ))
    return ret


def get_cmd(source, destination):
    return " ".join(
        [
            "ffmpeg",
            "-loglevel quiet",
            "-hide_banner",
            "-y",
            "-i",
            f'"{source}"',
            "-qscale:v 2",
            "-f image2",
            f'"{os.path.join(destination, "%04d.jpg")}"',
        ]
    )


def extract_frames(task):
    fgr_video_pathname, pha_video_pathname, fgr_frame_path, pha_frame_path = task
    os.makedirs(fgr_frame_path, exist_ok=True)
    os.makedirs(pha_frame_path, exist_ok=True)
    fgr_cmd = get_cmd(fgr_video_pathname, fgr_frame_path)
    os.system(fgr_cmd)
    pha_cmd = get_cmd(pha_video_pathname, pha_frame_path)
    os.system(pha_cmd)


if __name__ == '__main__':
    task_list = get_task_list()
    thread_map(extract_frames, task_list)
