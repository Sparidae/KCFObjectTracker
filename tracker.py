import os

import cv2
import numpy as np


class Tracker:
    def __init__(self) -> None:
        # HOGparams
        self.HOGparams = {
            "winSize": None,
            "blockSize": (8, 8),
            "blockStride": (4, 4),
            "cellSize": (4, 4),
            "nbins": 9,
        }

        pass

    def init(self, frame, bbox):
        # 使用框定的frame和bbox初始化tracker
        # 初始化hog特征提取器
        self.hog = cv2.HOGDescriptor(**self.HOGparams)
        # 生成HOG特征图x
        x = self._gen_feature()
        # 使用二维高斯函数生成训练标签矩阵y
        y = self._g2()
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

    def _train(self, x, y, sigma, lambda_):
        # 通过原论文实现的训练函数
        # 返回核化岭回归的解alphaf
        # TODO 实现DCF
        k = self._rbf_kernel_correlation(x, x, sigma)
        alphaf = np.fft.fft2(y) / (np.fft.fft2(k) + lambda_)
        return alphaf

    def _rbf_kernel_correlation(self, x1, x2, sigma):
        c = np.fft.ifft2(np.sum(np.conj(np.fft.fft2(x1)) * np.fft.fft2(x2), axis=0))
        c = np.fft.fftshift(c)  # 将零频率分量设置为频谱中心
        d = np.sum(x1**2) + np.sum(x2**2) - 2 * c
        k = np.exp(-1 / (sigma**2) * np.abs(d) / d.size)
        return k

    def _detect(self, alphaf, x, z, sigma):
        k = self._rbf_kernel_correlation(z, x, sigma)
        responses = np.real(np.fft.ifft2(alphaf * np.fft.fft2(k)))
        return responses


if __name__ == "__main__":
    t = Tracker()
    img = cv2.imread(".\\OTB100\\Basketball\\img\\0001.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t._hog(img)
