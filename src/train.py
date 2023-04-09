from typing import Tuple

from torchreid.reid.models.osnet import OSNet, init_pretrained_weights, OSBlock
from torchreid.reid.data import ImageDataManager
from torchreid.reid.utils import check_isfile, load_pretrained_weights
from torchreid.reid.data.datasets import register_image_dataset
from torchreid.reid.optim import build_lr_scheduler, build_optimizer

import os
import hydra
import pyrootutils
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.vtx import VTX
from src.data.vtx_mini import VTX_MINI
from src.data.vtx_test import VTX_TEST


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
            config_name="train.yaml")
def train(cfg: DictConfig) -> Tuple[dict, dict]:
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

    output_dir = os.path.join(cfg.paths.log_dir, cfg.output_dir)
    pretrained = (cfg.model_path and check_isfile(cfg.model_path))

    model = osnet_x1_0(num_classes=datamanager.num_train_pids,
                       pretrained=not pretrained,
                       loss=cfg.model.loss,
                       feature_dim=cfg.model.feature_dim,
                       use_gpu=cfg.model.use_gpu)

    if pretrained:
        load_pretrained_weights(model, cfg.model_path)

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
    print(datamanager.transform_tr)
    print('pretrained: ', pretrained)
    print('model_path: ', cfg.model_path)
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
               rerank=cfg.rerank,
               fixbase_epoch=cfg.fixbase_epoch,
               open_layers=cfg.open_layers)

    print('+++++++++++++++')
    print(datamanager.transform_tr)
    print('pretrained: ', pretrained)
    print('model_path: ', cfg.model_path)
    print(model.feature_dim)
    print('+++++++++++++++')


if __name__ == "__main__":
    train()