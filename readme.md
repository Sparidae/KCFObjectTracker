## KCF目标跟踪


使用KCF实现的目标跟踪系统

## 文件结构

```text
dataset/
    OTB100/ # OTB数据集，由视频的图片序列组成
        xxx/
        ...
    video.avi # 视频文件数据

tracker.py # KCF实现
main.py # 主要实验
```


## 准备

1. 测试数据集
2. 读取数据：
    - 数据集读取：img序列+rect文本
    - 视频文件：video.xxx
    - 实时视频：cv读取视频设备画面


## 参考

[KCF目标跟踪方法分析与总结](https://www.cnblogs.com/YiXiaoZhou/p/5925019.html)
