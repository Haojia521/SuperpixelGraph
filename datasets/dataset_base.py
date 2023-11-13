import os.path as osp
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __init__(self, root, split, transform=None):
        super(DatasetBase, self).__init__()

        self.root = root
        self.split = split
        self.data_dir = osp.join(self.root, split)
        self.index = self.collect_data_index()

        self.transform = transform

    def collect_data_index(self):
        raise NotImplementedError

    def read_image(self, idx):
        raise NotImplementedError

    def read_label(self, idx):
        raise NotImplementedError

    def get_mean_value(self):
        raise NotImplementedError

    def __getitem__(self, item):
        idx = self.index[item]

        img = self.read_image(idx)
        gt = self.read_label(idx)

        if self.transform is not None:
            img, gt = self.transform(img, gt)

        img_mean_value = self.get_mean_value()
        if img_mean_value is not None:
            img = img - img_mean_value

        return img, gt

    def __len__(self):
        return len(self.index)
