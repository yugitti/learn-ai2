from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import sampler
from torchvision import datasets
from torchvision import transforms
import numpy as np
from PIL import Image


class SimpleDataset(Dataset):

    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        img = self.x[index]
        img = img.view([1, 28, 28])
        if self.transform is not None:
            img = self.transform(img)
        target = self.y[index]
        return img, target

    def __len__(self):
        return len(self.x)


class InfiniteSampler(sampler.Sampler):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        while True:
            order = np.random.permutation(self.num_samples)
            for i in range(self.num_samples):
                yield order[i]

    def __len__(self):
        return None


def get_iters(
        dataset='mnist', root_path='.', data_transforms=None,
        n_labeled=100, valid_size=1000,
        l_batch_size=32, ul_batch_size=128, test_batch_size=256,
        workers=8, pseudo_label=None):

    # train_path = '../../pytorch/data/'
    # test_path = '../../pytorch/data/'

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root=root_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root_path, train=False, download=False, transform=transform)

    if data_transforms is None:
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
            'eval': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
        }

    # print(train_dataset.train_data)

    x_train, y_train = train_dataset.train_data, np.array(train_dataset.train_labels)
    x_test, y_test = test_dataset.test_data, np.array(test_dataset.test_labels)
    #
    labeled_idx = []
    # for i in range(10):
    #     labeled_idx[i] = np.where(y_train==i)[0][:n_labeled//10]
    # x_train = np.delete(x_train, labeled_idx, axis=0)
    #
    # randperm = np.random.permutation(len(x_train))
    # validation_idx = randperm[:valid_size]
    # unlabeled_idx = randperm[valid_size:]

    randperm = np.random.permutation(len(x_train))
    y_train_tmp = y_train[randperm]
    labeled_det = []
    each_label_num = n_labeled // 10
    for i in range(10):
        labeled_det = np.hstack((labeled_det, np.where(y_train_tmp==i)[0][:each_label_num])).astype(np.int)

    labeled_idx = randperm[labeled_det]
    randperm = np.delete(randperm, labeled_det)

    validation_idx = randperm[:valid_size]
    unlabeled_idx = randperm[valid_size:]


    # randperm = np.random.permutation(len(x_train))
    # labeled_idx = randperm[:n_labeled]
    # validation_idx = randperm[n_labeled:n_labeled + valid_size]
    # unlabeled_idx = randperm[n_labeled + valid_size:]

    x_labeled = x_train[labeled_idx]
    x_validation = x_train[validation_idx]
    x_unlabeled = x_train[unlabeled_idx]

    y_labeled = y_train[labeled_idx]
    y_validation = y_train[validation_idx]


    print('label_data:   {}'.format(len(labeled_idx)))
    print('unlabel_data: {}'.format(len(unlabeled_idx)))
    print('valid_data:   {}'.format(len(validation_idx)))
    print('labeled_idx: {}'.format(labeled_idx))
    print('y_labeled: {}'.format(y_labeled))


    if pseudo_label is None:
        y_unlabeled = y_train[unlabeled_idx]
    else:
        assert isinstance(pseudo_label, np.ndarray)
        y_unlabeled = pseudo_label

    data_iterators = {
        'labeled': iter(DataLoader(
            SimpleDataset(x_labeled, y_labeled, data_transforms['train']),
            batch_size=l_batch_size, num_workers=workers,
            sampler=InfiniteSampler(len(x_labeled)),
        )),
        'unlabeled': iter(DataLoader(
            SimpleDataset(x_unlabeled, y_unlabeled, data_transforms['train']),
            batch_size=ul_batch_size, num_workers=workers,
            sampler=InfiniteSampler(len(x_unlabeled)),
        )),
        'make_pl': iter(DataLoader(
            SimpleDataset(x_unlabeled, y_unlabeled, data_transforms['eval']),
            batch_size=ul_batch_size, num_workers=workers, shuffle=False
        )),
        'val': iter(DataLoader(
            SimpleDataset(x_validation, y_validation, data_transforms['eval']),
            batch_size=len(x_validation), num_workers=workers, shuffle=False
        )),
        'test': DataLoader(
            SimpleDataset(x_test, y_test, data_transforms['eval']),
            batch_size=test_batch_size, num_workers=workers, shuffle=False
        )
    }

    return data_iterators