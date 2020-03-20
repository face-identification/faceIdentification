import argparse
import numpy as np
import random
import time
import cv2
import face_recognition as fr

__author__ = 'Ivan Wang'

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=2000, help="minimum area size")
args = vars(ap.parse_args())

# 如果video参数为None，那么我们从摄像头读取数据
videoUrl = args["video"]
min_area = args["min_area"]
if args.get("video", None) is None:
    camera = cv2.VideoCapture("./test.mp4")
# 否则我们读取一个视频文件
else:
    camera = cv2.VideoCapture(videoUrl)

time.sleep(2)

# 初始化全局变量，背景帧
bgFrame = None
last_total_area = 0
diff_frame_count = 0

def detection(frame):
    box = fr.face_locations(frame)
    face = frame[box[0][0]:box[0][2],box[0][3]:box[0][1]]
    return face

def random_bright(img, p=0.5, lower=0.5, upper=1.5):
    if random.random() < p:
        mean = np.mean(img)
        img = img - mean
        img = img * random.uniform(lower, upper) + mean * random.uniform(lower, upper)  # 亮度
        img = img / 255.
    return img

index=1
num = 1
# 遍历视频的每一帧
while True:
    # 获取当前帧并初始化occupied/unoccupied文本，1秒24帧
    (grabbed, frame) = camera.read()
    # 如果不能抓取到一帧，说明我们到了视频的结尾
    if not grabbed:
        break
    if index % 5 != 0:
        index += 1
        continue
    face = detection(frame)
    img = random_bright(face, 1)
    cv2.imwrite("./faces/"+str(num)+".jpg",face)
    num += 1
    cv2.imwrite("./faces/"+str(num)+".jpg",img*255)
    #cv2.imshow("", img)
    #cv2.waitKey(0)
    num += 1
    index+=1

# 清理摄像机资源并关闭打开的窗口
camera.release()
cv2.destroyAllWindows()