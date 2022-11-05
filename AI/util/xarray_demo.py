import xarray as xr
import numpy as np

# 全球数据
input_data = r'/Users/whvixd/Documents/individual/MODIS/dataset/gpp/2018/NIRv.GPP.201801.v1.nc'  #数据存放路径
data = xr.open_dataset(input_data)                                   #使用xarray读取数据
print(data)

gpp = data['GPP']
# gpp_data_array=xr.DataArray(gpp.data, coords=[lat_grid, lon_grid], dims=['lat', 'lon'])

# 重采样 中国经纬度数据
lon_grid25 = np.arange(73.66, 135.05, 0.05) # 0.05度 5.5km
lat_grid25 = np.arange(3.86, 53.55, 0.05)

# 双线性插值，网格到网格
tm_grid25 = gpp.interp(longitude=lon_grid25,latitude=lat_grid25, method='linear')

print(tm_grid25.data.shape) # (1228, 994)
print(tm_grid25.data)
