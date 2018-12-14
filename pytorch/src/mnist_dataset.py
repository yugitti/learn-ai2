from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import pandas as pd
import os

class MNISTDataSet(Dataset):
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
        label = self.image_dataframe.loc[idx, 'label']

        if self.transform:
            image = self.transform(image)

        return image, label