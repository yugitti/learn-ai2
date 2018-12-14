import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.models.resnet
import torchvision.datasets as dset
import torchvision.transforms as transforms
import dataset

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import easydict

import net
import run
from logger import TrainLogger

def main():
    args = easydict.EasyDict({
        # "dataroot": "/mnt/gold/users/s18150/mywork/pytorch/data/gan",
        "dataroot": "/mnt/gold/users/s18150/mywork/pytorch/data",
        "save_dir": "./",
        "prefix": "test",
        "workers": 8,
        "batch_size": 128,
        "image_size": 32,
        # "image_size": 28,
        # "nc": 3,
        "nc": 1,
        "nz": 100,
        "ngf": 32,
        "ndf": 32,
        # "ngf": 28,
        # "ndf": 64,
        "epochs": 1,
        "lr": 0.0002,
        "beta1": 0.5,
        "gpu": 7,
        "use_cuda": True,
        "feature_matching": True,
        "mini_batch": True,
        "iters": 50000,
        "label_batch_size": 100,
        "unlabel_batch_size": 100,
        "test_batch_size": 10,
        "out_dir": './result',
        "log_interval": 500,
        "label_num": 20

    })

    manualSeed = 999
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    device = torch.device('cuda:{}'.format(args.gpu) if args.use_cuda else 'cpu')

    # transform = transforms.Compose([
    #     transforms.Resize(args.image_size),
    #     transforms.CenterCrop(args.image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    #
    # dataset = dset.ImageFolder(root=args.dataroot, transform=transform)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
    #                                          shuffle=True, num_workers=args.workers)


    data_iterators = dataset.get_iters(
        root_path=args.dataroot,
        l_batch_size=args.label_batch_size,
        ul_batch_size=args.unlabel_batch_size,
        test_batch_size=args.test_batch_size,
        workers=args.workers,
        n_labeled=args.label_num
    )

    trainloader_label = data_iterators['labeled']
    trainloader_unlabel = data_iterators['unlabeled']
    testloader = data_iterators['test']


    # Generator用のモデルのインスタンス作成
    netG = net.Generator(args.nz, args.ngf, args.nc).to(device)
    # Generator用のモデルの初期値を設定
    netG.apply(net.weights_init)

    # Discriminator用のモデルのインスタンス作成
    netD = net.Discriminator(args.nc, args.ndf, device, args.batch_size, args.mini_batch).to(device)
    # Discriminator用のモデルの初期値を設定
    netD.apply(net.weights_init)

    # BCE Loss classのインスタンスを作成
    criterionD = nn.CrossEntropyLoss()
    # criterionD = nn.BCELoss()

    if args.feature_matching is True:
        criterionG = nn.MSELoss(reduction='elementwise_mean')
    else:
        criterionG = nn.BCELoss()

    # Generatorに入力するノイズをバッチごとに作成 (バッチ数は64)
    # これはGeneratorの結果を描画するために使用する
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # 最適化関数のインスタンスを作成
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    logger = TrainLogger(args)
    r = run.NNRun(netD, netG, optimizerD, optimizerG, criterionD, criterionG, device, fixed_noise, logger,args)

    # 学習
    # r.train(dataloader)
    r.train(trainloader_label, trainloader_unlabel, testloader)


if __name__ == '__main__':

    main()
#