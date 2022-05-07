
# 依赖 操作系统中的 gdal 软件，mac需要 brew install gdal
try:
    import osgeo.gdal as gdal
except ImportError:
    try:
        import gdal
    except ImportError:
        raise 'Python GDAL library not found, please install python-gdal'

try:
    import osgeo.osr as osr
except ImportError:
    try:
        import osr
    except ImportError:
        raise 'Python GDAL library not found, please install python-gdal'

try:
    import pymodis
except ImportError:
    try:
        import pymodis
    except ImportError:
        raise 'PyModis library not found, please install pyModis'
