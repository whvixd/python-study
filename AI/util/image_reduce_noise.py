from matplotlib import pyplot as plt  # 展示图片
import numpy as np  # 数值处理

import cv2  # opencv库


# from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 回归分析

def normalization(image):
    """
    将数据线性归一化
    :param image: 图片矩阵，一般是np.array 类型
    :return: 将归一化后的数据，在（0,1）之间
    """
    # 获取图片数据类型对象的最大值和最小值
    info = np.iinfo(image.dtype)

    # 图像数组数据放缩在 0-1 之间
    return image.astype(np.double) / info.max


def save_image(filename, image):
    """
    将np.ndarray 图像矩阵保存为一张 png 或 jpg 等格式的图片
    :param filename: 图片保存路径及图片名称和格式
    :param image: 图像矩阵，一般为np.array
    :return:
    """
    # np.copy() 函数创建一个副本。
    # 对副本数据进行修改，不会影响到原始数据，它们物理内存不在同一位置。
    img = np.copy(image)

    # 从给定数组的形状中删除一维的条目
    img = img.squeeze()

    # 将图片数据存储类型改为 np.uint8
    if img.dtype == np.double:
        # 若img数据存储类型是 np.double ,则转化为 np.uint8 形式
        img = img * np.iinfo(np.uint8).max

        # 转换图片数组数据类型
        img = img.astype(np.uint8)

    # 将 RGB 方式转换为 BGR 方式
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 生成图片
    cv2.imwrite(filename, img)


def plot_image(image, image_title, is_axis=False):
    """
    展示图像
    :param image: 展示的图像，一般是 np.array 类型
    :param image_title: 展示图像的名称
    :param is_axis: 是否需要关闭坐标轴，默认展示坐标轴
    :return:
    """
    # 展示图片
    plt.imshow(image)

    # 关闭坐标轴,默认关闭
    if not is_axis:
        plt.axis('off')

    # 展示受损图片的名称
    plt.title(image_title)

    # 展示图片
    plt.show()


def read_image(img_path):
    """
    读取图片，图片是以 np.array 类型存储
    :param img_path: 图片的路径以及名称
    :return: img np.array 类型存储
    """
    # 读取图片
    img = cv2.imread(img_path)

    # 如果图片是三通道，采用 matplotlib 展示图像时需要先转换通道
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def restore_image(noise_img, size=4):
    """
    使用 你最擅长的算法模型 进行图像恢复。
    :param noise_img: 一个受损的图像
    :param size: 输入区域半径，长宽是以 size*size 方形区域获取区域, 默认是 4
    :return: res_img 恢复后的图片，图像矩阵值 0-1 之间，数据类型为 np.array,
            数据类型对象 (dtype): np.double, 图像形状:(height,width,channel), 通道(channel) 顺序为RGB
    """
    # 恢复图片初始化，首先 copy 受损图片，然后预测噪声点的坐标后作为返回值。
    res_img = np.copy(noise_img)

    # 获取噪声图像
    noise_mask = get_noise_mask(noise_img)
    # -------------实现图像恢复代码答题区域----------------------------
    for i in range(noise_mask.shape[0]):
        for j in range(noise_mask.shape[1]):
            for k in range(noise_mask.shape[2]):
                if noise_mask[i, j, k] == 0:
                    sc = 1
                    listx = get_window_small(res_img, noise_mask, i, j, k)
                    if len(listx) != 0:
                        res_img[i, j, k] = listx[len(listx) // 2]
                    else:
                        while (len(listx) == 0):
                            listx = get_window(res_img, noise_mask, sc, i, j, k)
                            sc = sc + 1
                        res_img[i, j, k] = listx[len(listx) // 2]

    # ---------------------------------------------------------------

    return res_img


def noise_mask_image(img, noise_ratio):
    """
    根据题目要求生成受损图片
    :param img: 图像矩阵，一般为 np.ndarray
    :param noise_ratio: 噪声比率，可能值是0.4/0.6/0.8
    :return: noise_img 受损图片, 图像矩阵值 0-1 之间，数据类型为 np.array,
             数据类型对象 (dtype): np.double, 图像形状:(height,width,channel),通道(channel) 顺序为RGB
    """
    # 受损图片初始化
    noise_img = None

    # -------------实现受损图像答题区域-----------------
    import random
    print(img.shape)
    noise_img = np.copy(img)
    for i in range(3):
        for j in range(img.shape[0]):
            mask = list(range(img.shape[1]))
            mask = random.sample(mask, int(img.shape[1] * noise_ratio))
            for k in range(img.shape[1]):
                if k in mask:
                    noise_img[j, k, i] = 0

    # -----------------------------------------------

    return noise_img


def get_noise_mask(noise_img):
    """
    获取噪声图像，一般为 np.array
    :param noise_img: 带有噪声的图片
    :return: 噪声图像矩阵
    """
    # 将图片数据矩阵只包含 0和1,如果不能等于 0 则就是 1。
    return np.array(noise_img != 0, dtype='double')


def get_window(res_img, noise_mask, sc, i, j, k):
    listx = []

    if i - sc >= 0:
        starti = i - sc
    else:
        starti = 0
    if j + 1 <= res_img.shape[1] - 1 and noise_mask[0, j + 1, k] != 0:
        listx.append(res_img[0, j + 1, k])
    if j - 1 >= 0 and noise_mask[0, j - 1, k] != 0:
        listx.append(res_img[0, j - 1, k])

    if i + sc <= res_img.shape[0] - 1:
        endi = i + sc
    else:
        endi = res_img.shape[0] - 1
    if j + 1 <= res_img.shape[1] - 1 and noise_mask[endi, j + 1, k] != 0:
        listx.append(res_img[endi, j + 1, k])
    if j - 1 >= 0 and noise_mask[endi, j - 1, k] != 0:
        listx.append(res_img[endi, j - 1, k])

    if j + sc <= res_img.shape[1] - 1:
        endj = j + sc
    else:
        endj = res_img.shape[1] - 1
    if i + 1 <= res_img.shape[0] - 1 and noise_mask[i + 1, endj, k] != 0:
        listx.append(res_img[i + 1, endj, k])
    if i - 1 >= 0 and noise_mask[i - 1, endj, k] != 0:
        listx.append(res_img[i - 1, endj, k])

    if j - sc >= 0:
        startj = j - sc
    else:
        startj = 0
    if i + 1 <= res_img.shape[0] - 1 and noise_mask[i + 1, 0, k] != 0:
        listx.append(res_img[i + 1, 0, k])
    if i - 1 >= 0 and noise_mask[i - 1, 0, k] != 0:
        listx.append(res_img[i - 1, 0, k])

    for m in range(starti, endi + 1):
        for n in range(startj, endj + 1):
            if noise_mask[m, n, k] != 0:
                listx.append(res_img[m, n, k])
    listx.sort()
    return listx


def get_window_small(res_img, noise_mask, i, j, k):
    listx = []
    sc = 1
    if i - sc >= 0 and noise_mask[i - 1, j, k] != 0:
        listx.append(res_img[i - 1, j, k])

    if i + sc <= res_img.shape[0] - 1 and noise_mask[i + 1, j, k] != 0:
        listx.append(res_img[i + 1, j, k])

    if j + sc <= res_img.shape[1] - 1 and noise_mask[i, j + 1, k] != 0:
        listx.append(res_img[i, j + 1, k])

    if j - sc >= 0 and noise_mask[i, j - 1, k] != 0:
        listx.append(res_img[i, j - 1, k])
    listx.sort()
    return listx


def restore_image(noise_img, size=4):
    """
    使用 你最擅长的算法模型 进行图像恢复。
    :param noise_img: 一个受损的图像
    :param size: 输入区域半径，长宽是以 size*size 方形区域获取区域, 默认是 4
    :return: res_img 恢复后的图片，图像矩阵值 0-1 之间，数据类型为 np.array,
            数据类型对象 (dtype): np.double, 图像形状:(height,width,channel), 通道(channel) 顺序为RGB
    """
    # 恢复图片初始化，首先 copy 受损图片，然后预测噪声点的坐标后作为返回值。
    res_img = np.copy(noise_img)

    # 获取噪声图像
    noise_mask = get_noise_mask(noise_img)
    # -------------实现图像恢复代码答题区域----------------------------
    for i in range(noise_mask.shape[0]):
        for j in range(noise_mask.shape[1]):
            for k in range(noise_mask.shape[2]):
                if noise_mask[i, j, k] == 0:
                    sc = 1
                    listx = get_window_small(res_img, noise_mask, i, j, k)
                    if len(listx) != 0:
                        res_img[i, j, k] = listx[len(listx) // 2]
                    else:
                        while (len(listx) == 0):
                            listx = get_window(res_img, noise_mask, sc, i, j, k)
                            sc = sc + 1
                        res_img[i, j, k] = listx[len(listx) // 2]

    # ---------------------------------------------------------------

    return res_img


if __name__ == '__main__':
    '''
    图像去噪，参考：
    https://github.com/yunwei37/ZJU-CS-GIS-ClassNotes/blob/master/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD/image-restoration/main.ipynb
    
    '''

    # 原始图片
    # 加载图片的路径和名称
    img_path = '/Users/whvixd/Downloads/A.png'

    # 读取原始图片
    img = read_image(img_path)

    # 展示原始图片
    plot_image(image=img, image_title="original image")

    # 生成受损图片
    # 图像数据归一化
    nor_img = normalization(img)

    # 噪声比率
    noise_ratio = 0.6

    # 生成受损图片
    noise_img = noise_mask_image(nor_img, noise_ratio)

    if noise_img is not None:
        # 展示受损图片
        plot_image(image=noise_img, image_title="the noise_ratio = %s of original image" % noise_ratio)

        # 恢复图片
        res_img = restore_image(noise_img)

        # 计算恢复图片与原始图片的误差
        # print("恢复图片与原始图片的评估误差: ", compute_error(res_img, nor_img))
        # print("恢复图片与原始图片的 SSIM 相似度: ", calc_ssim(res_img, nor_img))
        # print("恢复图片与原始图片的 Cosine 相似度: ", calc_csim(res_img, nor_img))

        # 展示恢复图片
        plot_image(image=res_img, image_title="restore image")

        # 保存恢复图片
        save_image('res_' + img_path, res_img)
    else:
        # 未生成受损图片
        print("返回值是 None, 请生成受损图片并返回!")
