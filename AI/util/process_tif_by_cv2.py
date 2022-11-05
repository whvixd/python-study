import cv2
img = cv2.imread("/Users/whvixd/Documents/individual/MODIS/dataset/gpp/2018/tifs/GPP_1.tif",1)

from osgeo import gdal
import numpy as np


def load_img(path):
    dataset = gdal.Open(path)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    im_data = im_data.transpose((1, 2, 0))  # 此步保证矩阵为channel_last模式
    return im_data

if __name__ == '__main__':
    tif_1=load_img("/Users/whvixd/Documents/individual/MODIS/dataset/gpp/2018/tifs/GPP_1.tif")
    print(tif_1)
