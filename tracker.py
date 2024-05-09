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
        # HOG参数
        self.block_size = (8, 8)
        self.block_stride = (4, 4)
        self.cell_size = (4, 4)
        self.n_bins = 9

        # 将ROI放缩到这个大小再提取HOG特征
        self.max_patch_size = 256

        #
        self.padding = 2.5  # 扩大ROI的倍数，帮助模型获取目标周围环境信息
        self.sigma = 0  # FIXME
        self.lambda_ = 0  #
        pass

    def init(self, frame, roi):
        # 使用框定的frame和bbox初始化tracker，frame是当前图像帧
        x, y, w, h = roi  # 左上角表示的方法

        # 计算等比例的将ROI最长边缩放到maxpatchsize的大小，该大小需能被blockstride整除
        _s = self.max_patch_size / max(w, h)
        self._hog_win_size = (int(w * _s) // 4 * 4 + 4, int(h * _s) // 4 * 4 + 4)
        self._init_hog(self._hog_win_size)

        # 生成HOG特征图
        x = self._gen_feature(frame, roi)

        # 使用二维高斯函数生成训练标签矩阵y
        y = self._gen_label(x.shape[1:])  # 二三维是特征维度
        # 计算kxx  训练非线性回归器得到alphah
        self.alphaf = self._train(x, y, self.sigma, self.labmda_)

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

    def _init_hog(self, win_size):
        # 初始化HOG特征提取器
        # win_size是需要提取的图像区域的大小，该大小是等比例的将ROI最长边缩放到maxpatchsize的大小
        self.hog = cv2.HOGDescriptor(
            win_size,
            self.block_size,
            self.block_stride,
            self.cell_size,
            self.n_bins,
        )
        return

    def _gen_feature(self, image, roi):
        # 计算目标区域的HOG特征
        # 1. 计算2.5x扩大的ROI，用于获取目标环境信息

        # 2.裁剪出扩大后的ROI的子图片，并将这个图片resize到之前计算的hog win size，用于计算特征

        # 3. 计算特征，返回值为36,h,w
        feature = self.hog.compute()
        _, feat_h, feat_w = feature.shape

        # 4. 针对特征信号通过hanning窗进行平滑处理，减少泄漏
        hann2d = np.hanning(feat_h).reshape(-1, 1) * np.hanning(feat_w)
        feature = feature * hann2d
        return feature  # 36,h,w

    def _gen_label(self, w, h):
        # 计算二维高斯作为特征标签
        # 1. 生成坐标矩阵，并纠正网格坐标的偏移
        halfw, halfh = w / 2 - 0.5, h / 2 - 0.5
        x, y = np.mgrid[-halfw : halfw + 0.5 : 1, -halfh : halfh + 0.5 : 1]  # 坐标矩阵
        # 2. 得到用于计算二维高斯的sigma
        # 参考matlab源码
        # https://github.com/scott89/KCF/blob/master/gaussian_shaped_labels.m
        # https://github.com/joaofaro/KCFcpp/blob/master/src/kcftracker.cpp
        output_sigma = 0.125
        sigma = np.sqrt(w * h) / self.padding * output_sigma  # FIXME
        g = 1 / (2 * np.pi * sigma**2) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return g

    def _train(self, x, y, sigma, lambda_):
        # 通过原论文实现的训练函数
        # 返回核化岭回归的解alphaf
        # TODO 实现DCF
        k = self._rbf_kernel_correlation(x, x, sigma)
        alphaf = np.fft.fft2(y) / (np.fft.fft2(k) + lambda_)
        return alphaf

    def _rbf_kernel_correlation(self, x1, x2, sigma):
        # 径向基核相关函数，输入的多通道矩阵，比如转换为axis0为通道
        c = np.fft.ifft2(np.sum(np.conj(np.fft.fft2(x1)) * np.fft.fft2(x2), axis=0))
        c = np.fft.fftshift(c)  # 将零频率分量设置为频谱中心
        d = np.sum(x1**2) + np.sum(x2**2) - 2 * c
        k = np.exp(-1 / (sigma**2) * np.abs(d) / d.size)
        return k

    def _detect(self, alphaf, x, z, sigma):
        k = self._rbf_kernel_correlation(z, x, sigma)
        responses = np.real(np.fft.ifft2(alphaf * np.fft.fft2(k)))  # 逆变换后仍然是复数
        return responses


if __name__ == "__main__":
    t = Tracker()
    img = cv2.imread(".\\OTB100\\Basketball\\img\\0001.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    t._hog(img)
