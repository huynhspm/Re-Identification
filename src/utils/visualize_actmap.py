import numpy as np
import cv2
import torch
from torch.nn import functional as F
import os.path as osp

from torchreid.reid.utils import mkdir_if_missing

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