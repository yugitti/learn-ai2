from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import pandas as pd
import os

class Cifar10DataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):

        self.image_dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.image_dataframe.loc[idx, 'img'])
#         image = io.imread(img_name)
        image = Image.open(img_name)
        label = self.image_dataframe.loc[idx, 'label_index']

        if self.transform:
            image = self.transform(image)

        return image, label

def get_class_name(label):

    CIFAR10_LABEL_NAMES = (
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck')

    if label < len(CIFAR10_LABEL_NAMES):
        return CIFAR10_LABEL_NAMES[label]
    else:
        return 'invalid'
