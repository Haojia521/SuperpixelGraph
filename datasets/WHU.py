import numpy as np
import os.path as osp
from osgeo import gdal
import os
import cv2

from .dataset_base import DatasetBase


class DatasetWhu(DatasetBase):
    def __init__(self, root, split, transform=None):
        super(DatasetWhu, self).__init__(root, split, transform)

    def get_mean_value(self):
        return np.array([0.411, 0.432, 0.45]).reshape((3, 1, 1)).astype(np.float32)

    def collect_data_index(self):
        return [name[:-4] for name in os.listdir(osp.join(self.data_dir, 'image'))]

    def read_image(self, idx):
        dataset = gdal.Open(osp.join(self.data_dir, 'image', idx + '.tif'), gdal.GA_ReadOnly)
        band_arr_list = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
        img = np.dstack(band_arr_list)

        return img

    def read_label(self, idx):
        lbl = cv2.imread(osp.join(self.data_dir, 'label', idx + '.tif'), cv2.IMREAD_UNCHANGED) / 255
        return lbl
