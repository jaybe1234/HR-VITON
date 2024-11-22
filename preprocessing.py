from typing import List

import PIL
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from PIL.Image import Resampling
from torch import nn
from torchvision.transforms import transforms as T, InterpolationMode

from human_parsing_lip.net.pspnet import PSPNet
from pytorch_openpose.src.body import Body


def build_human_parsing_model():
    net = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet')
    state_dict = torch.load('./human_parsing_lip/checkpoints/densenet/PSPNet_last')
    net = nn.DataParallel(net)
    net.load_state_dict(state_dict)
    net = net.cuda().eval()
    return net

def build_openpose_model():
    return Body('pytorch_openpose/model/body_pose_model.pth')

def build_densepose_model():
    from densepose import add_densepose_config
    from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
    from densepose.vis.extractor import create_extractor

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    config_fpath = 'densepose-configs/densepose_rcnn_R_50_FPN_s1x.yaml'
    model_fpath = './checkpoints/densepose_rcnn_R_50_FPN_s1x.pkl'
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    cfg.merge_from_list(['MODEL.ROI_HEADS.SCORE_THRESH_TEST', '0.8'])
    cfg.MODEL.WEIGHTS = model_fpath
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    visualizer = DensePoseResultsFineSegmentationVisualizer(cfg=cfg, alpha=1)
    extractor = create_extractor(visualizer)

    return predictor, visualizer, extractor


def denormalize(img: np.ndarray, mean: List[float], std: List[float]):
    c, _, _ = img.shape
    for idx in range(c):
        img[idx, :, :] = img[idx, :, :] * std[idx] + mean[idx]
    return img


def generate_human_parsing(image: PIL.Image, size=(192, 256)):
    human_parsing_model = build_human_parsing_model()
    transforms = T.Compose([
        T.Resize((256, 256), 3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = transforms(image).cuda()
    with torch.no_grad():
        pred, _ = human_parsing_model(image.unsqueeze(0))
    pred = pred.squeeze(dim=0)
    pred = pred.cpu().numpy().transpose(1, 2, 0)
    pred = np.argmax(pred, axis=2).astype(np.uint8)
    pred = Image.fromarray(pred, mode='L')
    pred = pred.resize(size, resample=Resampling.NEAREST)
    return pred


def generate_openpose(image: PIL.Image):
    openpose_model = build_openpose_model()
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    candidate, subset = openpose_model(image)
    key = candidate[subset[0][:18].astype(np.int8)][:, :2]
    key[subset[0][:18] == -1] = 0
    return key

def get_im_parse_agnostic(im_parse, pose_data, w=768, h=1024) -> Image:
    parse_array = np.array(im_parse)
    parse_upper = ((parse_array == 5).astype(np.float32) +
                   (parse_array == 6).astype(np.float32) +
                   (parse_array == 7).astype(np.float32))
    parse_neck = (parse_array == 10).astype(np.float32)

    r = 10
    agnostic = im_parse.copy()

    # mask arms
    for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
        mask_arm = Image.new('L', (w, h), 'black')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        i_prev = pose_ids[0]
        for i in pose_ids[1:]:
            if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
            pointx, pointy = pose_data[i]
            radius = r*4 if i == pose_ids[-1] else r*15
            mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
            i_prev = i
        parse_arm = (np.array(mask_arm).astype(np.float32) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    # mask torso & neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

    return agnostic

def generate_densepose(img: PIL.Image):
    predictor, visualizer, extractor = build_densepose_model()
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    with torch.no_grad():
        outputs = predictor(img)
    data = extractor(outputs['instances'])
    out = np.zeros_like(img)
    out = visualizer.visualize(out, data)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out)

def get_agnostic(im, im_parse, pose_data):
    parse_array = np.array(im_parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                  (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                   (parse_array == 12).astype(np.float32) +
                   (parse_array == 16).astype(np.float32) +
                   (parse_array == 17).astype(np.float32) +
                   (parse_array == 18).astype(np.float32) +
                   (parse_array == 19).astype(np.float32))

    agnostic = im.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[8] - pose_data[11])
    point = (pose_data[8] + pose_data[11]) / 2
    pose_data[8] = point + (pose_data[8] - point) / length_b * length_a
    pose_data[11] = point + (pose_data[11] - point) / length_b * length_a

    r = int(length_a / 16) + 1

    # mask torso
    for i in [8, 11]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 8]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 11]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [8, 11]], 'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 11, 8]], 'gray', 'gray')

    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*5, pointy-r*9, pointx+r*5, pointy), 'gray', 'gray')

    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*12)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'gray', 'gray')
    for i in [3, 4, 6, 7]:
        if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

    for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
        # mask_arm = Image.new('L', (self.fine_width, self.fine_height), 'white')
        mask_arm = Image.new('L', (768, 1024), 'white')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        pointx, pointy = pose_data[pose_ids[0]]
        mask_arm_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'black', 'black')
        for i in pose_ids[1:]:
            if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r*10)
            pointx, pointy = pose_data[i]
            if i != pose_ids[-1]:
                mask_arm_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'black', 'black')
        mask_arm_draw.ellipse((pointx-r*4, pointy-r*4, pointx+r*4, pointy+r*4), 'black', 'black')

        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
    return agnostic


def get_uppers(cloth_path: str, cloth_mask_path: str):
    cloth = Image.open(cloth_path).convert('RGB')
    cloth_mask = Image.open(cloth_mask_path)
    return cloth, cloth_mask


def get_dataset(img_path: str, cloth_path: str, mask_path: str):

    img = Image.open(img_path)
    key = generate_openpose(img)
    human_parse = generate_human_parsing(img, img.size)
    densepose = generate_densepose(img)
    cloth_agnostic = get_im_parse_agnostic(human_parse, key)
    agnostic = get_agnostic(img, human_parse, key)
    cloth, cloth_mask = get_uppers(cloth_path, mask_path)

    # Convert to tensor
    transform = T.Compose([ \
        T.ToTensor(), \
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    labels = {
        0:  ['background',  [0, 10]],
        1:  ['hair',        [1, 2]],
        2:  ['face',        [4, 13]],
        3:  ['upper',       [5, 6, 7]],
        4:  ['bottom',      [9, 12]],
        5:  ['left_arm',    [14]],
        6:  ['right_arm',   [15]],
        7:  ['left_leg',    [16]],
        8:  ['right_leg',   [17]],
        9:  ['left_shoe',   [18]],
        10: ['right_shoe',  [19]],
        11: ['socks',       [8]],
        12: ['noise',       [3, 11]]
    }

    ## densepose
    densepose = transform(densepose)

    ## cloth
    cloth = transform(cloth)

    ## cloth mask
    cloth_mask = (np.array(cloth_mask) >= 128).astype(np.float32)
    cloth_mask = torch.from_numpy(cloth_mask).unsqueeze(0)

    ## cloth agnostic
    cloth_agnostic = torch.from_numpy(np.array(cloth_agnostic)).long().unsqueeze_(0)
    parse_map = torch.zeros((20, 1024, 768), dtype=torch.float32)
    parse_map = parse_map.scatter_(0, cloth_agnostic, 1.0)

    cloth_agnostic_oh = torch.zeros((13, 1024, 768))
    for i in range(len(labels)):
        for label in labels[i][1]:
            cloth_agnostic_oh[i] += parse_map[label]

    ## human agnostic
    human_agnostic = get_agnostic(img, human_parse, key)
    human_agnostic = transform(human_agnostic)

    return {
        "cloth": cloth,
        "cloth_mask": cloth_mask,
        "human_agnostic": human_agnostic,
        "densepose": densepose,
        "cloth_agnostic": cloth_agnostic_oh,
    }


if __name__ == '__main__':
    img_path = './data/train/image/00000_00.jpg'
    cloth_path = './data/train/cloth/00000_00.jpg'
    mask_path = './data/train/mask/00000_00.jpg'
    get_dataset(img_path, cloth_path, mask_path)