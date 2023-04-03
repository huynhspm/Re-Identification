from typing import Tuple

from torchreid.reid.models.osnet import OSNet, init_pretrained_weights, OSBlock
from torchreid.reid.utils import (check_isfile, load_pretrained_weights)

import hydra
import pyrootutils
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


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
def inference(cfg: DictConfig) -> Tuple[dict, dict]:
    # model_path = 'logs/osnet_x1_0_from_scratch_full_data_filter_bboxes/model.pth.tar-5'
    pretrained = (cfg.model_path and check_isfile(cfg.model_path))

    model = osnet_x1_0(num_classes=1,
                       pretrained=not pretrained,
                       loss=cfg.model.loss,
                       feature_dim=cfg.model.feature_dim,
                       use_gpu=cfg.model.use_gpu)

    print('+++++++++++++++')
    print('pretrained: ', pretrained)
    print('model_path: ', cfg.model_path)
    print(model.feature_dim)
    print('+++++++++++++++')

    if pretrained:
        load_pretrained_weights(model, cfg.model_path)

    model.eval()
    model = model.cuda()

    import torch
    import glob
    import cv2
    from torchvision.transforms import Resize, Compose, ToTensor, Normalize

    transform = Compose([
        ToTensor(),
        Resize((256, 128)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    images = []

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    images_path = glob.glob('data/*.jpg')
    images_path.sort()
    for image_path in images_path:
        image = cv2.imread(image_path)
        image = image[..., ::-1]
        image = transform(image.copy())

        images.append(image)
        print(image_path, '     ', image.min(), image.max())
        img = image.clone().moveaxis(0, -1)
        img = (img * std + mean) * 255
        img = img.numpy()
        img = img[..., ::-1]
        # cv2.imwrite(image_path.split('/')[-1], img)

    images = torch.stack(images, dim=0)

    print(images.shape)
    features = model(images)
    print(features.min(), features.max())
    print(features.shape)
    print(torch.cdist(features, features))


if __name__ == "__main__":
    inference()