from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import torchgeometry as tgm
from torchvision.utils import save_image

import numpy as np

from network_generator import SPADEGenerator
from networks import ConditionGenerator
import os

from preprocessing import get_dataset



@dataclass
class ModelOption:
    gpu_id: int = 0
    warp_feature: str = "T1"
    out_layer: str = "relu"
    cuda: bool = True
    num_upsampling_layers: str = "most"
    fine_height: int = 1024
    fine_width: int = 768
    ngf: int = 64
    norm_G: str = "spectralaliasinstance"
    gen_semantic_nc: int = 7


def load_checkpoint(model, checkpoint_path, opt):
    if not os.path.exists(checkpoint_path):
        print('no checkpoint')
        raise
    log = model.load_state_dict(torch.load(checkpoint_path), strict=False)
    if opt.cuda :
        model.cuda()

def load_checkpoint_G(model, checkpoint_path,opt):
    if not os.path.exists(checkpoint_path):
        print("Invalid path!")
        return
    state_dict = torch.load(checkpoint_path)
    new_state_dict = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict.items()])
    new_state_dict._metadata = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict._metadata.items()])
    model.load_state_dict(new_state_dict, strict=True)
    if opt.cuda :
        model.cuda()


opt = ModelOption()
tocg = ConditionGenerator(opt, input1_nc=4, input2_nc=16, output_nc=13, ngf=96, norm_layer=nn.BatchNorm2d)
load_checkpoint(tocg, 'mtviton.pth', opt)
tocg.eval()

generator = SPADEGenerator(opt, 9)
generator.eval()
load_checkpoint(generator, 'gen.pth', opt)

def make_grid(N, iH, iW,opt):
    grid_x = torch.linspace(-1.0, 1.0, iW).view(1, 1, iW, 1).expand(N, iH, -1, -1)
    grid_y = torch.linspace(-1.0, 1.0, iH).view(1, iH, 1, 1).expand(N, -1, iW, -1)
    if opt.cuda :
        grid = torch.cat([grid_x, grid_y], 3).cuda()
    else:
        grid = torch.cat([grid_x, grid_y], 3)
    return grid

def remove_overlap(seg_out, warped_cm):

    assert len(warped_cm.shape) == 4

    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm


def inference(dataset):
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    pre_cloth = dataset["cloth"].unsqueeze(0).cuda()
    cloth_down = F.interpolate(pre_cloth, (256, 192), mode="bilinear")
    pre_cloth_mask = dataset["cloth_mask"].unsqueeze(0).cuda()
    cloth_mask_down = F.interpolate(pre_cloth_mask, size=(256, 192), mode='nearest')

    cloth_agnostic = dataset["cloth_agnostic"].unsqueeze(0).cuda()
    cloth_agnostic_down = F.interpolate(cloth_agnostic, size=(256, 192), mode='nearest')
    densepose = dataset["densepose"].unsqueeze(0).cuda()
    densepose_down = F.interpolate(densepose, size=(256, 192), mode='bilinear')

    human_agnostic = dataset["human_agnostic"].unsqueeze(0).cuda()

    input1 = torch.cat([cloth_down, cloth_mask_down], dim=1).cuda()
    input2 = torch.cat([cloth_agnostic_down, densepose_down], dim=1).cuda()
    flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt,input1, input2)

    cloth_mask = torch.ones_like(fake_segmap)
    cloth_mask[:,3:4, :, :] = warped_clothmask_paired
    fake_segmap = fake_segmap * cloth_mask

    fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(opt.fine_height, opt.fine_width), mode='bilinear'))
    fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

    old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
    labels = {
        0:  ['background',  [0]],
        1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
        2:  ['upper',       [3]],
        3:  ['hair',        [1]],
        4:  ['left_arm',    [5]],
        5:  ['right_arm',   [6]],
        6:  ['noise',       [12]]
    }

    parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
    for i in range(len(labels)):
        for label in labels[i][1]:
            parse[:, i] += old_parse[:, label]

    # warped cloth
    N, _, iH, iW = pre_cloth.shape
    flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
    flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
    grid = make_grid(N, iH, iW,opt)
    warped_grid = grid + flow_norm
    warped_cloth = F.grid_sample(pre_cloth, warped_grid, padding_mode='border')
    warped_clothmask = F.grid_sample(pre_cloth_mask, warped_grid, padding_mode='border')

    # remove occlusion
    warped_clothmask = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask)
    warped_cloth = warped_cloth * warped_clothmask + torch.ones_like(warped_cloth) * (1-warped_clothmask)

    return generator(torch.cat((human_agnostic, densepose, warped_cloth), dim=1), parse)

if __name__ == '__main__':
    # generator.print_network()
    img_path = "./data/test/image/05006_00.jpg"
    cloth_path = "./data/test/cloth/11001_00.jpg"
    cloth_mask_path = "./data/test/cloth-mask/11001_00.jpg"

    dataset = get_dataset(img_path, cloth_path, cloth_mask_path)
    output = inference(dataset)
    img = output[0].cpu()/2 + 0.5
    save_image(img, 'output_test.png')

    print("done")
