import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import easydict

import net
import run


def main():
    args = easydict.EasyDict({
        "dataroot": "/mnt/gold/users/s18150/mywork/pytorch/data/gan",
        "save_dir": "./",
        "prefix": "minibach_feature",
        "workers": 8,
        "batch_size": 128,
        "image_size": 64,
        "nc": 3,
        "nz": 100,
        "ngf": 64,
        "ndf": 64,
        "epochs": 50,
        "lr": 0.0002,
        "beta1": 0.5,
        "gpu": 9,
        "use_cuda": True,
        "feature_matching": True,
        "mini_batch": True
    })

    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    device = torch.device('cuda:{}'.format(args.gpu) if args.use_cuda else 'cpu')

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = dset.ImageFolder(root=args.dataroot, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.workers)

    # Generator用のモデルのインスタンス作成
    netG = net.Generator(args.nz, args.ngf, args.nc).to(device)
    # Generator用のモデルの初期値を設定
    netG.apply(net.weights_init)

    # Discriminator用のモデルのインスタンス作成
    netD = net.Discriminator(args.nc, args.ndf, device, args.batch_size, args.mini_batch).to(device)
    # Discriminator用のモデルの初期値を設定
    netD.apply(net.weights_init)

    # BCE Loss classのインスタンスを作成
    criterionD = nn.BCELoss()

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

    r = run.NNRun(netD, netG, optimizerD, optimizerG, criterionD, criterionG, device, fixed_noise, args)

    # 学習
    r.train(dataloader)


if __name__ == '__main__':

    main()
#