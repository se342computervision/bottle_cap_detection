# bottle_cap_detection
在背景上随机放置10个瓶盖，检测各个瓶盖的位置和姿态。

## 目录
/bottle_cap_detection
    detect_caps.exec
    match.dat
    check_cap_open_direction.py
    color.py
    detection.py
    gui.py
    hog.py
    rotation.py
    sift.py
    sift_display.py


## 环境
python>=3.6
opencv_python=4.1.2
numpy
PIL
labelme
selectivesearch
skimage
json
pickle


## 运行
### 运行exec可执行文件：
在 detect_caps 所在目录中运行，无需搭建环境
```bash
$ ./detect_caps
```

### 运行python脚本：
```bash
$ python gui.py
```

## 说明
点击 select images 选择多张图片，点击 detect all 开始瓶盖检测算法。检测完成后，会显示用颜色标记的瓶盖位置，方向和中心。点击 prev 和 next 可切换图片。
