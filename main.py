import argparse
import os
import random
import re
import time

import cv2

from tracker import Tracker

RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)


class TrackingTrial:
    def __init__(self, interval=10) -> None:
        self.interval = interval  # 等待时间间隔
        self.gt = True  # 是否绘制groundtruth
        self.video_type = None  # ['seq','video','camera']

    def read_seq(self, dataset_path="./dataset/OTB100"):
        # 1.读取图片序列
        video_dirs = []
        for entry in os.listdir(dataset_path):
            full_path = os.path.join(dataset_path, entry)
            if os.path.isdir(full_path):
                video_dirs.append(full_path)

        video_path = random.choice(video_dirs)
        # video_path = "./dataset/OTB100/Lemming"  # 适应性调整
        print(video_path)

        seq_path = os.path.join(video_path, "img/%04d.jpg")

        cap = cv2.VideoCapture(seq_path)
        # 读取groundtruth检测方框
        rects = []
        with open(os.path.join(video_path, "groundtruth_rect.txt"), "r") as f:
            for line in f:
                p = re.split(",|\t", line.strip())
                rects.append(tuple(int(num) for num in p))

        # rect_iter = iter(rects)
        # 返回cv2 videocapture对象和 groundtruth检测框
        self.video_type = "seq"
        return cap, rects

    def read_video(self, video_path):
        """在视频文件中读取

        Args:
            video_path (str): 视频文件的路径

        Returns:
            VideoCapture: cv2的VideoCapture对象
        """
        cap = cv2.VideoCapture(video_path)
        self.video_type = "video"
        return cap

    def read_camera(self, device=0):
        """从摄像头设备读取

        Args:
            device (int, optional): 从选定的设备编号读取. Defaults to 0.

        Returns:
            VideoCapture: cv2的VideoCapture对象
        """
        cap = cv2.VideoCapture(device)  # 读取设备0的图像
        self.video_type = "camera"
        return cap

    def show_ground_truth(self, cap, rects):
        assert self.video_type == "seq"
        rect_iter = iter(rects)
        # 计时
        curr = time.perf_counter()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Video Ended")
                break

            # 绘制标准方框
            x, y, bw, bh = next(rect_iter)  # groundtruth
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), GREEN, 1)

            # 在frame上绘制帧率
            prev = curr
            curr = time.perf_counter()
            fps = 1 / (curr - prev)
            cv2.putText(
                frame, f"{fps:.0f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE
            )

            # print(frame.shape)
            cv2.imshow(f"Ground Truth", frame)

            # 等待一段时间（等待针对图片序列或者视频），如果esc或者q就退出，针对使用摄像头设备的方法
            c = cv2.waitKey(self.interval) & 0xFF
            if c == 27 or c == ord("q"):
                print("Break")
                break

        cap.release()
        cv2.destroyAllWindows()

    def track_object(self, cap, rects=None, use_cv=False):
        # 确定已经读取
        # 如果是图片序列类型则rects非None
        assert self.video_type in ["seq", "video", "camera"]
        assert self.video_type != "seq" or rects is not None

        # 获得tracker
        tracker = None
        if use_cv:
            param = cv2.TrackerKCF.Params()
            param.detect_thresh = 0.3  # 0.5 降低这个值来提高跟踪的灵敏度
            param.sigma = 0.2  # 较大的值会使跟踪器对目标形状的变化更加不敏感
            param.lambda_ = 0.0001  # 较大的值会使跟踪器对目标形状的变化更加不敏感
            param.interp_factor = 0.075  # 0.075 调整这个值来快速适应
            param.output_sigma_factor = 0.1  # 1/16 提高检测大小的适应性，
            param.resize = True
            param.max_patch_size = 256  # 80
            tracker = cv2.TrackerKCF.create(parameters=param)
        else:
            # TODO 自己实现的KCF
            tracker = Tracker()

        status = 0  # 状态：初始化0，正在跟踪1
        curr = time.perf_counter()
        roi = None  # 初始方框
        # lbbox = None  # 上一个预测方框
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Video Ended")
                break

            success = True
            bbox = None
            if status == 0:
                # 获得初始ROI 返回
                if self.video_type == "seq":
                    # TODO 读取ROI ,第一个方框
                    roi = rects[0]
                    tracker.init(frame, roi)
                    status = 1  # 更改为跟踪状态
                    bbox = roi
                elif self.video_type == "video":
                    roi = cv2.selectROI("Tracking", frame, False, False)
                    tracker.init(frame, roi)
                    status = 1  # 更改为跟踪状态
                    bbox = roi
                    # 展示第一帧
                elif self.video_type == "camera":
                    # TODO 按键来停下视频一帧并框选图像，否则一直循环
                    self.interval = 1
                    if cv2.waitKey(self.interval) & 0xFF == ord("c"):
                        roi = cv2.selectROI("Tracking", frame, False, False)
                        tracker.init(frame, roi)
                        status = 1  # 更改为跟踪状态

            elif status == 1:
                success, bbox = tracker.update(frame)  # 是否成功和边界框
                if not success:
                    cv2.putText(frame, "Lost", (40, 20), 0, 0.5, RED, 2)

            # 绘制检测方框

            # if gt:  # groundtruth 方框 绿色
            #     x, y, bw, bh = next(rect_iter)  # groundtruth
            #     cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 1)

            if success and bbox is not None:  # 如果成功且检测到方框
                x, y, bw, bh = bbox
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), BLUE, 1)
            else:
                # TODO 丢失目标处理，重检测？
                pass

            # 在frame上绘制帧率
            prev = curr
            curr = time.perf_counter()
            fps = 1 / (curr - prev)
            cv2.putText(frame, f"{fps:.0f}", (5, 20), 0, 0.5, BLUE)
            # 展示图像
            cv2.imshow(f"Tracking", frame)

            # 等待一段时间（等待针对图片序列或者视频），如果esc或者q就退出，针对使用摄像头设备的方法
            c = cv2.waitKey(self.interval) & 0xFF  # 只取最后字节
            if c == 27 or c == ord("q"):
                print("Q break")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments")
    parser.add_argument(
        "data_type",
        choices=["seq", "video", "camera"],
        help="Data type: seq, video, or camera",
    )
    parser.add_argument(
        "optional_argument", nargs="?", default=None, help="Optional argument"
    )

    args = parser.parse_args()

    p = TrackingTrial(interval=10)
    if args.data_type == "seq":
        print("Data type is sequence")

        cap, rects = p.read_seq()
        p.track_object(cap, rects)

    elif args.data_type == "video":
        if args.optional_argument is None:
            print("Error: Video path is missing.")
        print("Data type is video, Path:", args.optional_argument)

        cap = p.read_video(video_path=args.optional_argument)
        p.track_object(cap)

    elif args.data_type == "camera":
        device_id = (
            int(args.optional_argument) if args.optional_argument is not None else 0
        )
        print("Data type is camera, Device ID:", device_id)

        cap = p.read_camera()
        p.track_object(cap)

    # cap = p.read_camera()

    # p.show_ground_truth(cap, rects)
    # p.track_object(cap, rects, True)  # opencv方法
