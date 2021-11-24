from pyhdf.SD import SD
import matplotlib.pyplot as plt
import numpy as np
import pprint

hdf = SD('/Users/xx/Downloads/MOD17A2H.A2021313.h30v06.006.2021322082525.hdf')
print(hdf.info())  # 信息类别数

data = hdf.datasets()

for idx, sds in enumerate(data.keys()):
    print(idx, sds)

# 数据获取
Lat = hdf.select('Gpp_500m')[:]
print(Lat)
# Lon = hdf.select('longitude')[:]
# print(Lon)

# 数据获取
# Map = hdf.select('cloud_fraction')
# data = Map.get()
# data = np.transpose(data)  # 将数据进行转置

# 变量信息读取与输出
# pprint.pprint(Map.attributes())
# 属性
# print(Map.attributes())

# 直接输出全部信息
# attr = Map.attributes(full=1)
# attNames = attr.keys()
# attNames.sort()
# print(attNames)
# fill_value = attr['_FILLVALUE'][0]
# scale_factor = attr['SCALE_FACTOR'][0]

for i in data:
    print(i)  # 具体类别
    img = hdf.select(i)[:]  # 图像数据
    # 数据，图像？？
    # print(hdf.select(i)[:].data.obj)
    # plt.imshow(img, cmap='gray')  # 显示图像
    # plt.show()
