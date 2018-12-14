
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import net
import run
import logger
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--gpu', default=1, type=int, help='gpu index')
    parser.add_argument('--p', default=0, help='use pretrained weight, 0: no, 1: yes')
    parser.add_argument('--n', default='temp', help='file prefix name')
    args = parser.parse_args()

    lr = args.lr
    gpu_id = args.gpu
    prefix = args.n

    batch_size = 100
    test_bach_size = 100
    epochs = 100
    lr = 0.01
    momentum = 0.5
    no_cuda = False
    seed = 123
    log_interval = 10
    out_dir = '../result'
    data_dir = '../data'
    test_interval = 1
    resume_interval = 1
    unit_num = 1000

    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    # device = torch.device('cuda:{}'.format(gpu_id) if use_cuda else 'cpu')
    device = torch.device('cuda:{}'.format(gpu_id))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    ## データの水増しと正規化
    transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print('start to load train dataset')
    trainset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, **kwargs
    )

    print('start to load test dataset')
    testset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_bach_size, shuffle=False, **kwargs
    )
    print('finish loading dataset')

    # net = net.Net().to(device)
    net = net.CNN().to(device)
    print('finish loading NN')

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(net.parameters())
    print('finish loading optimizer')

    criterion = nn.CrossEntropyLoss()
    print('finish loading criterion')

    logger = logger.TrainLogger(out_dir, epochs, prefix)
    print('finish loading logger')

    r = run.NNRun(net, optimizer, criterion, logger, device, log_interval, out_dir)

    print('start epoch loop')
    for epoch in range(1, epochs + 1):
        r.train(trainloader, epoch)
        if epoch % test_interval == 0:
            r.test(testloader)
            if logger.max_accuracy == logger.valid_accuracy[-1]:
                r.checkpoint(epoch)
