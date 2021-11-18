#!/usr/bin/env python3
from netCDF4 import Dataset
import numpy as np
import os

# panoply nc可视化工具：https://www.giss.nasa.gov/tools/panoply/download/

path = "/Users/didi/Downloads/12981977/2018/NIRv.GPP.201801.v1.nc"
dst = Dataset(path, mode='r', format="netCDF4")

dst.set_auto_mask(False)
# 查看nc文件中包含了什么
print(dst)
print('---------------------------------------------------------')
# 查看nc文件有哪些变量
print(dst.variables.keys())
print('--------------------------------------------------------')
# 查看nc文件中变量的属性名称
print(dst.variables.keys())
for i in dst.variables.keys():
    print('%s: %s' % (i, dst.variables[i].ncattrs()))
print('--------------------------------------------------------')
# 查看nc文件中变量的属性的具体信息
print(dst.variables.keys())
print('-------------------------latitude-------------------------------')
lat = dst.variables['latitude']
print(lat)
print(len(lat))
print(lat[:])
print('--------------------------longitude------------------------------')
lon = dst.variables['longitude']
print(lon)
print(len(lon))
print(lon[:])
print('--------------------------GPP------------------------------')
GPP = dst.variables['GPP']
print(GPP)
unit = GPP.units
print('--------------------------GPP[i][j]------------------------------')
for i in range(7200):
    for j in range(3600):
        val_ = GPP[i][j]
        if val_ != -9999 and val_ != 0:
            print("lon:{:},lat:{:},val_:{:} {:}".format(lon[i], lat[j], val_, unit))
