import torch

from model import MattingBase

if __name__ == '__main__':
    model = MattingBase('resnet50').cuda()
    true_fgr = torch.rand((4, 6, 3, 224, 224)).cuda()
    true_bgr = torch.rand((4, 6, 3, 224, 224)).cuda()
    true_pha = torch.rand((4, 6, 1, 224, 224)).cuda()
    true_src = true_bgr.clone()
    true_src = true_fgr * true_pha + true_src * (1 - true_pha)
    print(model(true_src).size())
