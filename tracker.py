import os

import cv2
import numpy as np


class Tracker:
    def __init__(self) -> None:
        pass

    def init(self, frame, bbox):
        # 使用框定的frame和bbox初始化tracker
        # 初始化hog特征提取器
        # 生成HOG特征图x
        # 使用二维高斯函数生成训练标签矩阵y
        # 计算kxx  训练非线性回归器得到alphah
        pass

    def update(self, frame):
        # 输入当前帧，输出bbox
        # 找到最大响应
        # 对不同尺度的候选区域
        # 1.提取候选区域特征z
        # 2.计算得到傅里叶域的响应矩阵
        # 3.找到最大响应和相关信息，比如移动坐标和变化尺度的wh（设置阈值，如果小于阈值认为目标丢失重新查找）

        # 更新目标区域信息、

        # 通过插值法更新模型 插值参数学习率m

        bbox = None
        return bbox

    def _hog(self, img):
        self.hog = cv2.HOGDescriptor()
        winStride = ()
        a = self.hog.compute(img, winStride=(576, 432))
        return a


if __name__ == "__main__":
    t = Tracker()
    img = cv2.imread(".\\OTB100\\Basketball\\img\\0001.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t._hog(img)
