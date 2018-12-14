import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from glob import glob
import os

import json
import numpy as np
from PIL import Image

import net
import run
import logger

if __name__ == '__main__':
    gpu = 5
    dir = '/public/dl_exp_cv_add/mini_cifar/train'

    temp =  os.path.join(dir, '*.png')
    paths = glob(temp)
    print(temp)
    assert len(paths) != 0
    if not os.path.exists('db'):
        os.mkdir('db')

    vgg16 = models.vgg16(pretrained=True)
    vgg16.eval()  ## 学習済みモデルを推論で使う場合には評価モードにする、そうしないと出力結果がその都度変わってしまう

    ## vgg16の学習で使用されたImageNetのデータをロードする
    ## 前処理としてvgg16で学習した時と同じ形式でデータを標準化する必要がある
    ## data: 0 ~ 1, サイズ 224 x 224
    ## 正規化 mean =  [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]


    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # test_image = paths[498]
    test_image = '../data/imagenet/n02423022_46.jpg'

    img = Image.open(test_image)
    img_tensor = preprocess(img)
    print(img_tensor.shape)

    ## バッチサイズを先頭に加えた 4次元のTensorに変換する
    img_tensor.unsqueeze_(0)
    print(img_tensor.shape)

    ## 画像をモデルへ入力
    out = vgg16(img_tensor)
    out_numpy = out.data.numpy()
    pred = np.argmax(out_numpy)
    sort = np.argsort(-out_numpy)[0]

    ## 予測結果を出力
    class_index = json.load(open('imagenet_class_index.json', 'r'))
    for i in range(5):
        print('predicted: {}'.format(class_index[str(sort[i])]))
