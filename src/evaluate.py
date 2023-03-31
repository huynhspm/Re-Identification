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
    register_image_dataset('aic2023', VTX)
    datamanager = ImageDataManager(root='data',
                                   sources='aic2023',
                                   height=256,
                                   width=128,
                                   train_sampler='RandomIdentitySampler',
                                   batch_size_train=5,
                                   batch_size_test=5)

    dir_path = 'logs/osnet_x1_0_from_scratch_full_data_filter_bboxes'
    model_path = dir_path + '/model.pth.tar-5'
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

    print('+++++++++++++++')
    print('pretrained: ', pretrained)
    print('model_path: ', model_path)
    print(model.feature_dim)
    print('+++++++++++++++')
    engine = ImageSoftmaxEngine(datamanager,
                                model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                use_gpu=True,
                                label_smooth=True)

    engine.run(save_dir=dir_path,
               max_epoch=5,
               start_epoch=0,
               print_freq=10,
               start_eval=0,
               eval_freq=10,
               test_only=True,
               dist_metric='euclidean',
               normalize_feature=False,
               visrank=True,
               visrank_topk=5,
               ranks=[1, 5],
               rerank=False)

    print('+++++++++++++++')
    print('pretrained: ', pretrained)
    print('model_path: ', model_path)
    print(model.feature_dim)
    print('+++++++++++++++')


if __name__ == "__main__":
    main()
