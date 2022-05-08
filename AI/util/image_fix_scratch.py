# 图像修复交互式案例——通过水流填充算法来修复被破坏的图像区域；
# 使用俩种方法进行修复
# cv2.INPAINT_TELEA （Fast Marching Method 快速行进算法），对位于点附近、边界法线附近和边界轮廓上的像素赋予更多权重。一旦一个像素被修复，它将使用快速行进的方法移动到下一个最近的像素。
# cv2.INPAINT_NS 流体力学算法，使用了流体力学的一些方法，基本原则是启发式的，首先沿着边从已知区域移动到未知区域（因为边是连续的）。它在匹配修复区域边界处的渐变向量的同时，继续等高线（连接具有相同强度的点的线，就像等高线连接具有相同高程的点一样）。

# USAGE 
# python image_fix_scratch.py ./image/scratch.png

# 按下鼠标左键,添加点、线，按下鼠标右键，添加矩形框，以制作被污染的需要修复图像
# 按下空格键：执行修复功能
# 按下r键：重置待修复的mask
# 按下esc键，退出

# 参考：https://www.jb51.net/article/219285.htm
import cv2
import numpy as np


class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None  # 线起始点
        self.drag_start = None  # 矩形起点
        self.drag_rect = None  # 矩形（左上角，右下角）坐标
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.drawing = False
        self.mode = False
        self.show()
        cv2.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv2.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
            self.drawing = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 第一次初始化时设定pt，往后保留上一个点作为矩形起点
            if self.drag_start == None:
                self.drag_start = pt

        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()

        if self.drag_start and flags & cv2.EVENT_FLAG_RBUTTON:
            xo, yo = self.drag_start
            x0, y0 = np.minimum([xo, yo], [x, y])
            x1, y1 = np.maximum([xo, yo], [x, y])
            self.drag_rect = None
            if x1 - x0 > 0 and y1 - y0 > 0:
                self.drag_rect = (x0, y0, x1, y1)
                for dst, color in zip(self.dests, self.colors_func()):
                    cv2.rectangle(dst, (x0, y0), (x1, y1), color, -1)
                self.dirty = True
                self.drag_start = None
                self.drag_rect = None
                self.show()
            else:
                self.drag_start = pt

    @property
    def dragging(self):
        return self.drag_rect is not None


def core():
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = '/Users/whvixd/Documents/PycharmProjects/workspace/python-study/AI/util/scratch.png'

    img = cv2.imread(fn)
    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    img_mark = img.copy()
    mark = np.zeros(img.shape[:2], np.uint8)
    sketch = Sketcher('img', [img_mark, mark], lambda: ((255, 255, 255), 255))

    while True:
        ch = cv2.waitKey()
        if ch == 27:
            break
        if ch == ord(' '):
            cv2.imshow('mask', mark)
            fmmres = cv2.inpaint(img_mark, mark, 3, cv2.INPAINT_TELEA)
            nsres = cv2.inpaint(img_mark, mark, 3, cv2.INPAINT_NS)
            cv2.imshow('inpaint fmm res', fmmres)
            cv2.imshow('inpaint ns res', nsres)
        if ch == ord('r'):
            img_mark[:] = img
            mark[:] = 0
            sketch.show()

    print('Done')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    core()
