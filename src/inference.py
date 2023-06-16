from typing import Tuple

from torchreid.reid.models.osnet import OSNet, init_pretrained_weights, OSBlock
from torchreid.reid.utils import (check_isfile, load_pretrained_weights)

import os.path as osp
import torch
import glob
import cv2
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

import hydra
import pyrootutils
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

transform = Compose([
    ToTensor(),
    Resize((256, 128)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
images = []


def osnet_x1_0(num_classes=1000,
               pretrained=True,
               loss='softmax',
               feature_dim=256,
               use_gpu=True):
    # standard size (width x1.0)
    model = OSNet(num_classes,
                  blocks=[OSBlock, OSBlock, OSBlock],
                  layers=[2, 2, 2],
                  channels=[64, 256, 384, 512],
                  loss=loss,
                  feature_dim=feature_dim,
                  use_gpu=use_gpu)
    if pretrained:
        init_pretrained_weights(model, key='osnet_x1_0')
    return model


def objective(thres_hold):
    loss = 0
    for image_1 in images:
        for image_2 in images:
            dist = (image_1['embedding'] - image_2['embedding'])**2
            dist = torch.sqrt(dist.sum())
            if dist == 0: continue

            if dist <= thres_hold and image_1['id'] == image_2['id']:
                loss += 1
            if dist > thres_hold and image_1['id'] != image_2['id']:
                loss += 1

    return -loss


def hyperopt_search():
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    trials = Trials()
    best = fmin(objective,
                space=hp.uniform('thres_hold', 1, 16),
                algo=tpe.suggest,
                max_evals=100,
                trials=trials)
    print(best)
    return best


def get_embedding_vector(model, image_dir):
    images_path = glob.glob(osp.join(image_dir, '*.jpg'))[:20]
    id = image_dir.split('/')[-1]

    for image_path in images_path:
        image = cv2.imread(image_path)
        image = image[..., ::-1]
        image = transform(image.copy())
        image = image.unsqueeze(0)
        embedding = model(image.to('cuda')).detach().cpu()
        image = image.detach().cpu()
        images.append({'id': id, 'embedding': embedding.squeeze()})
    return images


@hydra.main(version_base="1.3",
            config_path="../configs",
            config_name="eval.yaml")
def inference(cfg: DictConfig) -> Tuple[dict, dict]:

    output_dir = osp.join(cfg.paths.log_dir, cfg.output_dir)
    model_path = osp.join(output_dir, 'model', cfg.model_path)
    pretrained = (model_path and check_isfile(model_path))

    model = osnet_x1_0(num_classes=1,
                       pretrained=not pretrained,
                       loss=cfg.model.loss,
                       feature_dim=cfg.model.feature_dim,
                       use_gpu=cfg.model.use_gpu)

    print('+++++++++++++++')
    print('pretrained: ', pretrained)
    print('model_path: ', model_path)
    print(model.feature_dim)
    print('+++++++++++++++')

    if pretrained:
        load_pretrained_weights(model, model_path)

    model.eval()
    model = model.cuda()

    get_embedding_vector(model, 'data/vtx/gallery/8a_8_16')
    get_embedding_vector(model, 'data/vtx/gallery/8a_8_17')
    get_embedding_vector(model, 'data/vtx/gallery/8a_8_18')
    thres_hold = hyperopt_search()['thres_hold']

    right1 = 0
    wrong1 = 0
    right2 = 0
    wrong2 = 0
    for image_1 in images:
        for image_2 in images:
            dist = (image_1['embedding'] - image_2['embedding'])**2
            dist = torch.sqrt(dist.sum()).detach().cpu()
            if dist == 0: continue

            if image_1['id'] == image_2['id']:
                if dist <= thres_hold: right1 += 1
                else: wrong1 += 1
            else:
                if dist > thres_hold: right2 += 1
                else: wrong2 += 1

    print('+++++++++++++++')
    print('pretrained: ', pretrained)
    print('model_path: ', model_path)
    print(model.feature_dim)
    print('+++++++++++++++')
    print('num_images:', len(images))
    print('right1:', right1, 'wrong1:', wrong1, 'right2:', right2, 'wrong:',
          wrong2)

    # images = []
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    # images_path = glob.glob('data/inference/*.jpg')
    # images_path.sort()
    # for image_path in images_path:
    #     image = cv2.imread(image_path)
    #     image = image[..., ::-1]
    #     image = transform(image.copy())

    #     images.append(image)
    #     print(image_path, ' --- ', image.min(), ' --- ', image.max())
    #     img = image.clone().moveaxis(0, -1)
    #     img = (img * std + mean) * 255
    #     img = img.numpy()
    #     img = img[..., ::-1]
    #     # cv2.imwrite(image_path.split('/')[-1], img)

    # images = torch.stack(images, dim=0)

    # features = model(images.to('cuda'))
    # print(features.min(), features.max())
    # print(features.shape)
    # print(torch.cdist(features, features))


if __name__ == "__main__":
    inference()