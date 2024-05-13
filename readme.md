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


## 参考

[High-Speed Tracking with Kernelized Correlation Filters论文](https://ieeexplore.ieee.org/document/6870486?denied=)

[论文源码](https://github.com/chuanqi305/KCF)

[KCF目标跟踪方法分析与总结](https://www.cnblogs.com/YiXiaoZhou/p/5925019.html)

[[译]Kernelized Correlation Filters - KCF译文](https://zhuanlan.zhihu.com/p/55157416)

[【KCF算法解析】High-Speed Tracking with Kernelized Correlation Filters笔记](https://blog.csdn.net/EasonCcc/article/details/79658928)

## Usage

如果需要测试请将数据放在`./dataset`文件夹下，数据支持三种形式：
- 数据集读取：img序列+rect文本，如OTB数据集
- 视频文件：video.xxx
- 实时视频：cv读取视频设备画面


```shell
python main.py seq 
python main.py video [video_path]
python main.py camera [device_id]  
```