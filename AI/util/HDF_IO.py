from pyhdf.SD import SD
import matplotlib.pyplot as plt
import numpy as np
import pprint

hdf = SD('/Users/didi/Documents/whvixd/personal/studyAI/dataset/modis/MOD17A2H/MOD17A2H.A2021313.h21v03.006.2021322084625.hdf')
print(hdf.info())  # 信息类别数

data = hdf.datasets()

# for idx, sds in enumerate(data.keys()):
#     print(idx, sds)

# 数据获取
# Gpp_500m = hdf.select('Gpp_500m').get()
# print(Gpp_500m.shape)
# pprint.pprint(Gpp_500m.attributes())


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
    data = hdf.select(i).get()  # 数据
    print(str(i) + ".shape:", data.shape)
    # 数据，图像？？
    # print(hdf.select(i)[:].data.obj)
    # plt.imshow(img, cmap='gray')  # 显示图像
    # plt.show()
