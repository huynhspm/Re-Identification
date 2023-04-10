from typing import Tuple

from torchreid.reid.models.osnet import OSNet, init_pretrained_weights, OSBlock
from torchreid.reid.data import ImageDataManager
from torchreid.reid.utils import check_isfile, load_pretrained_weights
from torchreid.reid.data.datasets import register_image_dataset
from torchreid.reid.optim import build_lr_scheduler, build_optimizer
from torchreid.reid.utils import (check_isfile, mkdir_if_missing,
                                  load_pretrained_weights)

import numpy as np
import cv2
import torch
from torch.nn import functional as F
import os.path as osp

import hydra
import pyrootutils
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.vtx import VTX
from src.data.vtx_mini import VTX_MINI
from src.data.vtx_test import VTX_TEST

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10


@torch.no_grad()
def visactmap(model,
              test_loader,
              save_dir,
              width,
              height,
              use_gpu,
              img_mean=None,
              img_std=None):
    if img_mean is None or img_std is None:
        # use imagenet mean and std
        img_mean = IMAGENET_MEAN
        img_std = IMAGENET_STD

    model.eval()

    for target in list(test_loader.keys()):
        data_loader = test_loader[target]['query']  # only process query images
        # original images and activation maps are saved individually
        actmap_dir = osp.join(save_dir, 'actmap_' + target)
        mkdir_if_missing(actmap_dir)
        print('Visualizing activation maps for {} ...'.format(target))

        for batch_idx, data in enumerate(data_loader):
            imgs, paths = data['img'], data['impath']
            if use_gpu:
                imgs = imgs.cuda()

            # forward to get convolutional feature maps
            try:
                outputs = model(imgs, return_featuremaps=True)
            except TypeError:
                raise TypeError(
                    'forward() got unexpected keyword argument "return_featuremaps". '
                    'Please add return_featuremaps as an input argument to forward(). When '
                    'return_featuremaps=True, return feature maps only.')

            if outputs.dim() != 4:
                raise ValueError(
                    'The model output is supposed to have '
                    'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                    'Please make sure you set the model output at eval mode '
                    'to be the last convolutional feature maps'.format(
                        outputs.dim()))

            # compute activation maps
            outputs = (outputs**2).sum(1)
            b, h, w = outputs.size()
            outputs = outputs.view(b, h * w)
            outputs = F.normalize(outputs, p=2, dim=1)
            outputs = outputs.view(b, h, w)

            if use_gpu:
                imgs, outputs = imgs.cpu(), outputs.cpu()

            for j in range(outputs.size(0)):
                # get image name
                path = paths[j]
                imname = osp.basename(osp.splitext(path)[0])

                # RGB image
                img = imgs[j, ...]
                for t, m, s in zip(img, img_mean, img_std):
                    t.mul_(s).add_(m).clamp_(0, 1)
                img_np = np.uint8(np.floor(img.numpy() * 255))
                img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

                # activation map
                am = outputs[j, ...].numpy()
                am = cv2.resize(am, (width, height))
                am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) +
                                                1e-12)
                am = np.uint8(np.floor(am))
                am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

                # overlapped
                overlapped = img_np * 0.3 + am * 0.7
                overlapped[overlapped > 255] = 255
                overlapped = overlapped.astype(np.uint8)

                # save images in a single figure (add white spacing between images)
                # from left to right: original image, activation map, overlapped image
                grid_img = 255 * np.ones(
                    (height, 3 * width + 2 * GRID_SPACING, 3), dtype=np.uint8)
                grid_img[:, :width, :] = img_np[:, :, ::-1]
                grid_img[:,
                         width + GRID_SPACING:2 * width + GRID_SPACING, :] = am
                grid_img[:, 2 * width + 2 * GRID_SPACING:, :] = overlapped
                cv2.imwrite(osp.join(actmap_dir, imname + '.jpg'), grid_img)

            if (batch_idx + 1) % 10 == 0:
                print('- done batch {}/{}'.format(batch_idx + 1,
                                                  len(data_loader)))


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


@hydra.main(version_base="1.3",
            config_path="../configs",
            config_name="eval.yaml")
def eval(cfg: DictConfig) -> Tuple[dict, dict]:
    if cfg.data.sources == 'vtx':
        register_image_dataset('vtx', VTX)
    elif cfg.data.sources == 'vtx_mini':
        register_image_dataset('vtx_mini', VTX_MINI)
    elif cfg.data.sources == 'vtx_test':
        register_image_dataset('vtx_test', VTX_TEST)
    else:
        raise ValueError('Invalid dataset name')

    datamanager = ImageDataManager(root=cfg.data.root,
                                   sources=cfg.data.sources,
                                   height=cfg.data.height,
                                   width=cfg.data.width,
                                   train_sampler=cfg.data.train_sampler,
                                   batch_size_train=cfg.data.batch_size_train,
                                   batch_size_test=cfg.data.batch_size_test,
                                   transforms=list(cfg.data.transforms))

    output_dir = osp.join(cfg.paths.log_dir, cfg.output_dir)
    model_path = osp.join(output_dir, 'model', cfg.model_path)
    pretrained = (model_path and check_isfile(model_path))

    model = osnet_x1_0(num_classes=datamanager.num_train_pids,
                       pretrained=not pretrained,
                       loss=cfg.model.loss,
                       feature_dim=cfg.model.feature_dim,
                       use_gpu=cfg.model.use_gpu)

    if pretrained:
        load_pretrained_weights(model, model_path)

    if cfg.model.use_gpu:
        model = model.cuda()

    optimizer = build_optimizer(model,
                                optim=cfg.optimizer.optim,
                                lr=cfg.optimizer.lr)
    scheduler = build_lr_scheduler(optimizer,
                                   lr_scheduler=cfg.scheduler.lr_scheduler,
                                   stepsize=cfg.scheduler.stepsize)

    engine = hydra.utils.instantiate(cfg.engine,
                                     datamanager=datamanager,
                                     model=model,
                                     optimizer=optimizer,
                                     scheduler=scheduler)

    print('+++++++++++++++')
    print('pretrained: ', pretrained)
    print('model_path: ', model_path)
    print(model.feature_dim)
    print('+++++++++++++++')

    engine.run(save_dir=output_dir,
               max_epoch=cfg.max_epoch,
               start_epoch=cfg.start_epoch,
               print_freq=cfg.print_freq,
               start_eval=cfg.start_eval,
               eval_freq=cfg.eval_freq,
               test_only=cfg.test_only,
               dist_metric=cfg.dist_metric,
               normalize_feature=cfg.normalize_feature,
               visrank=cfg.visrank,
               visrank_topk=cfg.visrank_topk,
               ranks=cfg.ranks,
               rerank=cfg.rerank)

    if cfg.vis_actmap:
        print('visualize actmap')
        visactmap(model, datamanager.test_loader, output_dir, cfg.data.width,
                  cfg.data.height, cfg.model.use_gpu)


if __name__ == "__main__":
    eval()
