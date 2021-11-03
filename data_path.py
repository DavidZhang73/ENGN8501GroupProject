"""
Adapted from https://github.com/PeterL1n/BackgroundMattingV2
This file records the directory paths to the different datasets.
You will need to configure it for training the model.

All datasets follows the following format, where fgr and pha points to directory that contains jpg or png.
Inside the directory could be any nested formats, but fgr and pha structure must match. You can add your own
dataset to the list as long as it follows the format. 'fgr' should point to foreground images with RGB channels,
'pha' should point to alpha images with only 1 grey channel.
{
    'YOUR_DATASET': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        }
    }
}
"""

DATA_PATH = {
    'videomatte240k': {
        'train': r'<path to train>',
        'valid': r'<path to valid>'
    },
    'videomatte8k': {
        'train': r'<path to train>',
        'valid': r'<path to valid>'
    },
    'backgrounds': {
        'train': r'<path to train>',
        'valid': r'<path to valid>'
    },
}
