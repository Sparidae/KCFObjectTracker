import os
import random
import re
import time

import cv2

interval = 20

# 三种读取方式
# 1.读取图片序列
data_path = "./OTB100"
video_dirs = []
for entry in os.listdir(data_path):
    full_path = os.path.join(data_path, entry)
    if os.path.isdir(full_path):
        video_dirs.append(full_path)

video_path = random.choice(video_dirs)
seq_path = os.path.join(video_path, "img/%04d.jpg")
cap = cv2.VideoCapture(seq_path)
# 读取groundtruth检测方框
rects = []
with open(os.path.join(video_path, "groundtruth_rect.txt"), "r") as f:
    for line in f:
        p = re.split(",|\t", line.strip())
        rects.append(tuple(int(num) for num in p))
rect_iter = iter(rects)

# 2.实时摄像头
# 3.视频文件

# 计时
curr = time.perf_counter()

while cap.isOpened():
    pass
    ret, frame = cap.read()

    if not ret:
        print("Video Ended")
        break

    # 获得初始方框的方法
    # 1. 绘制初始方框
    # 2. 读取初始方框

    # 绘制检测方框
    # TODO 检测
    x, y, bw, bh = next(rect_iter)  # TODO groundtruth 替换为自行实现的KCF
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 1)

    # 在frame上绘制帧率
    prev = curr
    curr = time.perf_counter()
    fps = 1 / (curr - prev)
    cv2.putText(
        frame, f"{fps:.0f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
    )

    # 展示图像
    cv2.imshow("Tracking", frame)

    # 等待一段时间（等待针对图片序列或者视频），如果esc或者q就退出，针对使用摄像头设备的方法
    c = cv2.waitKey(interval) & 0xFF
    if c == 27 or c == ord("q"):
        print("Q break")
        break

cap.release()
cv2.destroyAllWindows()
