
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_dataset import MNISTDataSet
import net
import run
import logger


if __name__ == '__main__':


    batch_size = 100
    test_bach_size = 100
    epochs = 1
    lr = 0.01
    momentum = 0.5
    no_cuda = False
    gpu_id = 6
    seed = 123
    log_interval = 10
    out_dir = '../result'
    data_dir = '../data'
    test_interval = 1
    resume_interval = 1
    unit_num = 1000
    prefix = 'mnist'
    train_csv = '../data/mnist/train_big.csv'
    test_csv = '../data/mnist/test_big.csv'
    train_root_dir = '../data/mnist/train'
    test_root_dir = '../data/mnist/test'

    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device('cuda:{}'.format(gpu_id) if use_cuda else 'cpu')
    # device = torch.device('cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    ## 水増しとデータセットの正規化
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


    print('start to load train dataset')
    # trainset = datasets.MNIST(
    #     root=data_dir, train=True, download=True, transform=transform
    # )
    trainset = MNISTDataSet(train_csv, train_root_dir, transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, **kwargs
    )

    print('start to load test dataset')
    # testset = datasets.MNIST(
    #     root=data_dir, train=False, download=True, transform=transform
    # )
    testset = MNISTDataSet(test_csv, test_root_dir, transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_bach_size, shuffle=False, **kwargs
    )

    print('finish loading dataset')
    # net = net.Net().to(device)
    net = net.simpleCNN().to(device)

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(net.parameters())

    criterion = nn.CrossEntropyLoss()
    logger = logger.TrainLogger(out_dir, epochs, prefix)

    r = run.NNRun(net, optimizer, criterion, logger, device, log_interval, out_dir, prefix)

    for epoch in range(1, epochs + 1):
        r.train(trainloader, epoch)
        if epoch % test_interval == 0:
            r.test(testloader)
        if epoch % resume_interval == 0:
            r.checkpoint(epoch)
        logger.save_loss()