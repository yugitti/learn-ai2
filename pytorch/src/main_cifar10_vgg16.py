
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg16, vgg16_bn
from cifar10_dataset import Cifar10DataSet
import torch.nn.functional as F

import net
import run
import logger
import myvgg
import argparse

if __name__ == '__main__':
    # F.kl_div(0,0)

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('-g', default=9, type=int, help='gpu index')
    parser.add_argument('-n', default='test', help='file prefix name')
    parser.add_argument('-fg',  default=30, type=int, help='feature grad index')
    parser.add_argument('-cg',  default=0, type=int, help='classifier grad index')
    parser.add_argument('-b', default=40, type=int, help='batch size')

    args = parser.parse_args()

    lr = args.lr
    gpu_id = args.g
    prefix = args.n
    feature_grad = args.fg
    classifier_grad = args.cg
    batch_size = args.b

    test_bach_size = args.b
    epochs = 100
    lr = 0.01
    momentum = 0.5
    no_cuda = False
    # gpu_id = 0
    seed = 123
    log_interval = 10
    out_dir = '../result'
    data_dir = '../data'
    data_train_dir = '../data/cifar10_pictures/train'
    data_test_dir = '../data/cifar10_pictures/test'
    csv_train_file = '../data/cifar10_pictures/train_big.csv'
    csv_test_file = '../data/cifar10_pictures/test_big.csv'
    test_interval = 1
    resume_interval = 1
    unit_num = 1000



    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    # device = torch.device('cuda:{}'.format(gpu_id) if use_cuda else 'cpu')
    print('use cuda id [{}]'.format(gpu_id))
    # device = torch.device('cuda:{}'.format(gpu_id) if use_cuda else 'cpu')
    device = torch.device('cuda:{}'.format(gpu_id))

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    ## データの水増しと正規化
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    print('start to load train dataset')
    trainset = Cifar10DataSet(csv_train_file, data_train_dir, transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, **kwargs
    )

    # trainset = datasets.CIFAR10(
    #     root=data_dir, train=True, download=True, transform=transform
    # )
    # trainloader = torch.utils.data.DataLoader(
    #     trainset, batch_size=batch_size, shuffle=True, **kwargs
    # )



    print('start to load test dataset')
    testset = Cifar10DataSet(csv_test_file, data_test_dir, transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_bach_size, shuffle=False, **kwargs
    )
    # testset = datasets.CIFAR10(
    #     root=data_dir, train=False, download=True, transform=transform
    # )
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=test_bach_size, shuffle=False, **kwargs
    # )
    print('finish loading dataset')


    # vgg = vgg16(pretrained=True).to(device)

    print('use pretrained wieght')
    vgg = vgg16_bn(pretrained=True)
    net = net.VGG_New(vgg.features, vgg.classifier, device, num_classes=10, init_weights=False,
                      feature_grad=feature_grad, classifier_grad=classifier_grad)
    # net = net.myVGG16().to(device)
    # optimizer = optim.SGD(net.classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters())
    # net = net.myVGG16().to(device)
    # net = myvgg.VGG('VGG16').to(device)

    print('finish loading NN')

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    # optimizer = optim.Adam(net.parameters())
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    print('finish loading optimizer')

    criterion = nn.CrossEntropyLoss()
    print('finish loading criterion')

    logger = logger.TrainLogger(out_dir, epochs, args.n)
    print('finish loading logger')

    r = run.NNRun(net, optimizer, criterion, logger, device, log_interval, out_dir, prefix)

    print('start epoch loop')
    for epoch in range(1, epochs + 1):
        r.train(trainloader, epoch)
        if epoch % test_interval == 0:
            r.test(testloader)
            if logger.max_accuracy == logger.valid_accuracy[-1]:
                r.checkpoint(epoch)
            logger.save_loss()
