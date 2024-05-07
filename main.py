import os
import random
import re
import time

import cv2


class TrackingTrial:
    def __init__(self) -> None:
        self.interval = 15  # 等待时间间隔
        self.gt = True  # 是否绘制groundtruth

    def read_dataset(self):
        # 1.读取图片序列
        data_path = ".\\OTB100"
        video_dirs = []
        for entry in os.listdir(data_path):
            full_path = os.path.join(data_path, entry)
            if os.path.isdir(full_path):
                video_dirs.append(full_path)

        video_path = random.choice(video_dirs)
        video_path = ".\\OTB100\\Lemming"  # 适应性调整
        seq_path = os.path.join(video_path, "img/%04d.jpg")
        print(video_path)
        cap = cv2.VideoCapture(seq_path)
        # 读取groundtruth检测方框
        rects = []
        with open(os.path.join(video_path, "groundtruth_rect.txt"), "r") as f:
            for line in f:
                p = re.split(",|\t", line.strip())
                rects.append(tuple(int(num) for num in p))
        rect_iter = iter(rects)
        pass  # 返回cv2 videocapture对象和 groundtruth检测框
        return cap, rects

    def read_video(self):
        pass

    def read_camera(self):
        # 480*640*3
        cap = cv2.VideoCapture(0)  # 读取设备0的图像
        return cap

    def track_object(self, cap):
        use_cv = False
        # 获得tracker
        tracker = None
        if use_cv:
            # param = cv2.TrackerKCF.Params
            # param.detect_thresh = 0.3
            param = cv2.TrackerKCF.Params()
            param.detect_thresh = 0.3  # 0.5 降低这个值来提高跟踪的灵敏度
            param.sigma = 0.2  # 较大的值会使跟踪器对目标形状的变化更加不敏感
            param.lambda_ = 0.0001  # 较大的值会使跟踪器对目标形状的变化更加不敏感
            param.interp_factor = 0.075  # 0.075 调整这个值来快速适应
            param.output_sigma_factor = 1 / 16  # 1/16 提高检测大小的适应性，
            param.resize = True
            # param.max_patch_size = 300 # 80
            tracker = cv2.TrackerKCF.create(parameters=param)

        else:
            # TODO 自己实现的KCF

            pass
        status = "prepare"  # 状态：初始化0，正在跟踪1

        # 计时
        curr = time.perf_counter()
        init_frame = None
        lbbox = None  # 上一个预测方框
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Video Ended")
                break

            # 获得初始方框的方法
            # 1. 绘制初始方框
            # 2. 读取初始方框

            # 绘制检测方框
            # if gt:  # groundtruth 方框 绿色
            #     x, y, bw, bh = next(rect_iter)  # groundtruth
            #     cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
            # TODO 检测
            success = True
            bbox = None
            # if status == "prepare":
            #     init_frame = frame
            #     bbox = rects[0]
            #     tracker.init(init_frame, bbox)
            #     status = "tracking"
            # elif status == "tracking":
            #     success, bbox = tracker.update(frame)

            # if not success:
            #     # status = "lost"

            # elif status == "lost":
            #     pass

            # if success and bbox:
            #     x, y, bw, bh = bbox
            #     lbbox = bbox
            #     cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 1)
            # else:
            #     # tracker = cv2.TrackerKCF.create() # 使用前一帧进行初始化
            #     # tracker.init(frame, lbbox)
            #     # 使用最开始的进行初始化
            #     # tracker = cv2.TrackerKCF.create()
            #     # tracker.init(init_frame, rects[0])
            #     pass

            # 在frame上绘制帧率
            prev = curr
            curr = time.perf_counter()
            fps = 1 / (curr - prev)
            cv2.putText(
                frame,
                f"{fps:.0f}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

            # 展示图像
            # cv2.namedWindow("Tracking", 0)
            # cv2.resizeWindow("Tracking", 100)
            print(frame.shape)
            cv2.imshow(f"Tracking", frame)

            # 等待一段时间（等待针对图片序列或者视频），如果esc或者q就退出，针对使用摄像头设备的方法
            c = cv2.waitKey(self.interval) & 0xFF
            if c == 27 or c == ord("q"):
                print("Q break")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    p = TrackingTrial()
    # cap = p.read_camera()
    cap, rects = p.read_dataset()
    p.track_object(cap)
