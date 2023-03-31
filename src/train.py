from vtx import VTX
from torchreid.reid.models.osnet import OSNet, init_pretrained_weights, OSBlock
from torchreid.reid.data import ImageDataManager
from torchreid.reid.engine import ImageTripletEngine, ImageSoftmaxEngine
from torchreid.reid.utils import check_isfile, load_pretrained_weights
from torchreid.reid.data.datasets import register_image_dataset
from torchreid.reid.optim import build_lr_scheduler, build_optimizer

register_image_dataset('vtx', VTX)


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


def main():
    datamanager = ImageDataManager(root='data',
                                   sources='vtx',
                                   height=256,
                                   width=128,
                                   train_sampler='RandomIdentitySampler',
                                   batch_size_train=32,
                                   batch_size_test=100,
                                   transforms=['random_flip', 'random_erase'])

    dir_path = 'logs/osnet_x1_0_origin_market_fine_tune'
    model_path = 'logs/osnet_x1_0_origin_market/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
    pretrained = (model_path and check_isfile(model_path))

    model = osnet_x1_0(num_classes=datamanager.num_train_pids,
                       pretrained=not pretrained,
                       loss='triplet',
                       feature_dim=512,
                       use_gpu=True)
    if pretrained:
        load_pretrained_weights(model, model_path)

    model = model.cuda()
    optimizer = build_optimizer(model, optim="adam", lr=0.0003)
    scheduler = build_lr_scheduler(optimizer,
                                   lr_scheduler="single_step",
                                   stepsize=20)

    # frozen first layer
    for param in model.conv1.parameters():
        param.requires_grad = False

    engine = ImageTripletEngine(datamanager,
                                model,
                                optimizer=optimizer,
                                margin=0.3,
                                weight_t=1,
                                weight_x=1,
                                scheduler=scheduler,
                                use_gpu=True,
                                label_smooth=True)

    # engine = ImageSoftmaxEngine(datamanager,
    #                             model,
    #                             optimizer=optimizer,
    #                             scheduler=scheduler,
    #                             use_gpu=True,
    #                             label_smooth=True)

    print('+++++++++++++++')
    print(datamanager.transform_tr)
    print('pretrained: ', pretrained)
    print('model_path: ', model_path)
    print(model.feature_dim)
    print('+++++++++++++++')

    engine.run(save_dir=dir_path,
               max_epoch=5,
               start_epoch=0,
               print_freq=10,
               start_eval=0,
               eval_freq=10,
               test_only=False,
               dist_metric='euclidean',
               normalize_feature=False,
               visrank=False,
               visrank_topk=10,
               ranks=[1, 5, 10, 20],
               rerank=False)

    print('+++++++++++++++')
    print(datamanager.transform_tr)
    print('pretrained: ', pretrained)
    print('model_path: ', model_path)
    print(model.feature_dim)
    print('+++++++++++++++')


if __name__ == "__main__":
    main()