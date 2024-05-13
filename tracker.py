import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
命名约定
x,y,w,h = roi 是给定标准框ROI
ux,uy,uw,uh = uroi 是扩大2.5倍的ROI
fw,fh =self._hog_win_size 是计算特征图的大小
mw,mh 是用于提取特征的窗口大小
"""


class Tracker:
    def __init__(self) -> None:
        # 将ROI放缩到这个大小再提取HOG特征
        self.max_patch_size = 256

        # 可调整参数
        self.response_threshold = 0.15  # 小于目标maxres的这个比例就认为是丢失目标
        self.scale_factor = 0.05  # 调整方框大小每帧的变化率
        # 下列参数来自源代码https://github.com/scott89/KCF/blob/master/run_tracker.m
        self.padding = 2.5  # 扩大ROI的倍数，帮助模型获取目标周围环境信息
        self.kernel_sigma = 0.5  # 默认 0.5
        self.kernel_lambda_ = 1e-4  # 正则化参数 默认1e-4
        self.sigma_factor = 0.1  # 相对于目标大小的回归目标的空间带宽 默认0.1
        self.interp_factor = 0.005  # 更新率，更新alpha和x的速度 默认0.02

        # HOG参数
        self._win_size = None
        self.block_size = (8, 8)
        self.block_stride = (4, 4)
        self.cell_size = (4, 4)
        self.n_bins = 9

        # 不可调整参数
        self.target_maxres = None  # 最大的参数
        self.roi = None  # 存储的是未经缩放的roi
        self._hog_win_size = None  # 存储缩放为 maxpatchsize 的 窗口大小，提取特征的大小
        pass

    def init(self, frame, roi):
        # 使用框定的frame和bbox初始化tracker，frame是当前图像帧
        # 1. 得到区域大小
        _, _, w, h = roi

        # 2. 计算等比例的将ROI最长边缩放到maxpatchsize的大小，该大小需能被blockstride整除
        _factor = self.max_patch_size / max(w, h)
        self._hog_win_size = (
            round(w * _factor) // 4 * 4 + 4,
            round(h * _factor) // 4 * 4 + 4,
        )

        # 3. 初始化HOG特征提取器
        self.hog = cv2.HOGDescriptor(
            self._hog_win_size,
            self.block_size,
            self.block_stride,
            self.cell_size,
            self.n_bins,
        )

        # 4. 生成HOG特征图
        x, uroi = self._gen_feature(frame, roi)  # 返回形状 36,feat_h,feat_w

        # 5. 使用二维高斯函数生成训练标签矩阵y
        y = self._gen_label(x.shape[1:])  # y 的形状为h,w

        # 6. 计算kxx  训练非线性回归器得到alphaf
        self.alphaf = self._train(x, y)

        # %. 计算原始图像区域的最大响应
        response = self._detect(self.alphaf, x, x)
        self.target_maxres = np.max(response)
        print(self.target_maxres)
        print("-" * 10)

        self.roi = roi  # 模型当前的roi
        self.x = x  # 目标区域的特征图

    def update(self, frame):
        # 输入当前帧，输出新的roi。找到之前框定ROI中的最大响应
        max_record = -1
        for scale in [1 - self.scale_factor, 1, 1 + self.scale_factor]:
            # 0. 放缩roi
            x, y, w, h = self.roi
            roi_scaled = tuple(
                round(i)
                for i in (
                    x + w / 2 * (1 - scale),
                    y + h / 2 * (1 - scale),
                    w * scale,
                    h * scale,
                )
            )

            # 1.提取候选区域特征z
            z, uroi = self._gen_feature(frame, roi_scaled)
            ux, uy, uw, uh = uroi

            # 2.计算得到特征响应矩阵 返回形状为fh,fw
            responses = self._detect(self.alphaf, self.x, z)

            # 3.找到最大响应值和对应的中心坐标，如果是最大响应就更新变量
            # （设置阈值，如果小于阈值认为目标丢失重新查找）
            # 最大响应的值和下标
            max_res = np.max(responses)
            if max_res > max_record:
                max_record = max_res

                # 相对于uroi中心坐标的偏移
                y_maxres, x_maxres = np.unravel_index(
                    np.argmax(responses), responses.shape
                )
                dux = round(x_maxres * self.scale_ufw - (uw // 2))  # 减去中心坐标
                duy = round(y_maxres * self.scale_ufh - (uh // 2))

                x, y, w, h = roi_scaled
                roi_updated = (x + dux, y + duy, w, h)
                z_template = z

        # 4. 更新模型和信息，前提是根据判定确认追踪成功，否则认为目标lost
        self.roi = roi_updated
        # 通过插值法更新目标区域模板
        self.x = self.x * (1 - self.interp_factor) + z_template * self.interp_factor
        # 通过插值法更新模型
        y = self._gen_label(z_template.shape[1:])
        alphaf = self._train(z_template, y)
        self.alphaf = (
            self.alphaf * (1 - self.interp_factor) + alphaf * self.interp_factor
        )
        if max_record > self.target_maxres * self.response_threshold:
            print(max_record)
            # 更新roi信息
            # self.roi = roi_updated
            # # 通过插值法更新目标区域模板
            # self.x = self.x * (1 - self.interp_factor) + z_template * self.interp_factor
            # # 通过插值法更新模型
            # y = self._gen_label(z_template.shape[1:])
            # alphaf = self._train(z_template, y)
            # self.alphaf = (
            #     self.alphaf * (1 - self.interp_factor) + alphaf * self.interp_factor
            # )
            return True, self.roi
        else:
            print(max_record, " Lost")
            return False, None

    def _gen_feature(self, image, roi):
        """计算目标区域的HOG特征

        Args:
            image (_type_): 需要检测的帧
            roi (_type_): 目标区域ROI

        Returns:
            _type_: 特征矩阵 扩大后的ROI
        """
        # 1. 计算2.5x扩大的ROI，用于获取目标环境信息
        x, y, w, h = roi
        ux = round((x - w / 2 * (self.padding - 1)) // 2 * 2)
        uy = round((y - h / 2 * (self.padding - 1)) // 2 * 2)
        uw, uh = round(w * self.padding), round(h * self.padding)

        # 2. 裁剪+resize
        # 裁剪出2.5倍ROI的区域，并将这个图片resize到之前计算的hog win size，用于计算特征
        lx, rx = ux, ux + uw
        ly, ry = uy, uy + uh
        imh, imw = image.shape[:2]
        lx_pad = 0 if lx > 0 else 0 - lx
        ly_pad = 0 if ly > 0 else 0 - ly
        rx_pad = 0 if rx < imw else rx - imw
        ry_pad = 0 if ry < imh else ry - imh
        patch = image[uy if uy > 0 else 0 : uy + uh, ux if ux > 0 else 0 : ux + uw, :]
        patch = np.pad(patch, ((ly_pad, ry_pad), (lx_pad, rx_pad), (0, 0)), mode="edge")
        # plt.imshow(patch)
        # plt.colorbar()
        # plt.show()
        patch = cv2.resize(patch, self._hog_win_size)

        # 3. 计算特征，返回值为36,h,w 因为fft2默认计算最后两维
        feature = self.hog.compute(patch, self._hog_win_size, padding=(0, 0))
        mw, mh = self._hog_win_size  # 扩展为maxpatchsize的大小
        _o = (self.block_size[0] - self.block_stride[0]) // self.block_stride[0]
        feat_w = mw // self.block_stride[0] - _o  # 计算特征图宽高
        feat_h = mh // self.block_stride[1] - _o
        feature = feature.reshape(feat_w, feat_h, 36).transpose(2, 1, 0)
        # 计算 扩大roi比特征图大的倍数
        self.scale_ufw = uw / feat_w
        self.scale_ufh = uh / feat_h

        # 4. 针对特征信号通过hanning窗进行平滑处理，减少泄漏
        hann2d = np.hanning(feat_h).reshape(-1, 1) * np.hanning(feat_w)
        feature = feature * hann2d
        return feature, (ux, uy, uw, uh)  # 特征36,fh,fw  扩大后的roi

    def _gen_label(self, shape):
        """计算二维高斯作为特征标签

        Args:
            shape (tuple): (h,w) 高和宽，符合图像存储

        Returns:
            _type_: 形状为h,w的二维高斯标签矩阵
        """
        # 1. 生成坐标矩阵，并纠正网格坐标的偏移
        h, w = shape
        halfh, halfw = h / 2 - 0.5, w / 2 - 0.5
        y, x = np.mgrid[-halfh : halfh + 0.5 : 1, -halfw : halfw + 0.5 : 1]  # 坐标矩阵

        # 2. 得到用于计算二维高斯的sigma
        # 参考matlab源码得到计算sigma的公式
        # https://github.com/scott89/KCF/blob/master/gaussian_shaped_labels.m
        # https://github.com/scott89/KCF/blob/master/tracker.m
        sigma = np.sqrt(w * h) * self.sigma_factor / self.cell_size[0]
        g = 1 / (2 * np.pi * sigma**2) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return g

    def _train(self, x, y):
        """原论文的训练流程,返回核化岭回归的解alphaf

        Args:
            x (_type_): 形状 c,fh,fw
            y (_type_): fh,fw

        Returns:
            _type_: _description_
        """
        # TODO 实现DCF
        # TODO kernelsigma kernellambda_
        k = self._rbf_kernel_correlation(x, x)  # 形状为fh,fw
        alphaf = np.fft.fft2(y) / (np.fft.fft2(k) + self.kernel_lambda_)
        return alphaf

    def _rbf_kernel_correlation(self, x1, x2):
        """rbf径向基核相关函数

        Args:
            x1 (_type_): 输入形状 c,fh，fw 或者fh,fw
            x2 (_type_): 输入形状 c,fh，fw 或者fh,fw,至少有一个多通道的

        Returns:
            _type_: _description_
        """
        c = np.fft.ifft2(np.sum(np.conj(np.fft.fft2(x1)) * np.fft.fft2(x2), axis=0))
        c = np.fft.fftshift(c)  # 将零频率分量设置为频谱中心
        d = np.sum(x1**2) + np.sum(x2**2) - 2 * c
        k = np.exp(-1 / (self.kernel_sigma**2) * np.abs(d) / d.size)
        return k  # 形状fh,fw

    def _detect(self, alphaf, x, z):
        """检测区域，生成响应矩阵

        Args:
            alphaf (_type_): 模型岭回归的解
            x (_type_): _description_
            z (_type_): _description_

        Returns:
            _type_: 形状为fh,fw的特征响应矩阵
        """
        k = self._rbf_kernel_correlation(x, z)
        responses = np.real(np.fft.ifft2(alphaf * np.fft.fft2(k)))  # 逆变换后仍然是复数
        return responses  # 形状fh,fw


if __name__ == "__main__":
    t = Tracker()
    img = cv2.imread(".\\OTB100\\Basketball\\img\\0001.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
