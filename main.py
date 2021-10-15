import cv2
import torch
import numpy as np
import detector
import recognizer
import qx
import os
import dataTrain

yolo_dir = 'yolo_model'  # YOLO文件路径
weightsPath = os.path.join(yolo_dir, 'yolov3-voc_2000.weights')  # 权重文件
configPath = os.path.join(yolo_dir, 'yolov3-voc.cfg')  # 配置文件
labelsPath = os.path.join(yolo_dir, 'coco.names')  # label名称windows darknet python

CONFIDENCE = 0.5  # 过滤弱检测的最小概率
THRESHOLD = 0.4  # 非最大值抑制阈值

database = './FaceBase.db'
datasets = './datasets'
userInfo = {'stu_id': ''}

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
clicked = False

faceRecordCount = 0
flag = 0
stu_id = '00000'

q = qx.Queue()


if __name__ == '__main__':

    cameraCapture_in = cv2.VideoCapture(0)  # 打开编号为0或1的摄像头
    cameraCapture_in.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cameraCapture_in.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow('frame_in')  # 给视频框命名
    cv2.setMouseCallback('frame_in', detector.onMouse)
    print('入口显示摄像头，点击鼠标左键或按任意键退出')
    success_in, frame_in = cameraCapture_in.read()

    cameraCapture_out = cv2.VideoCapture(1)  # 打开编号为0或1的摄像头
    cameraCapture_out.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cameraCapture_out.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow('frame_out')  # 给视频框命名
    cv2.setMouseCallback('frame_out', detector.onMouse)
    print('显示出口摄像头，点击鼠标左键或按任意键退出')
    success_out, frame_out = cameraCapture_out.read()

    while success_in and success_out and cv2.waitKey(1) == -1 and not clicked:
        idxs,boxes,confidences,classIDs = detector.detect(net,frame_in)
        flag,faceRecordCount,stu_id = detector.show_frame(q,idxs, boxes, confidences, classIDs,flag,faceRecordCount,stu_id,frame_in)
        success_in, frame_in = cameraCapture_in.read()  # 摄像头获+++取下一帧

        if faceRecordCount != 0:
            dataTrain.train()

        idxs, boxes, confidences, classIDs =detector.detect(net, frame_out)
        recognizer.show_frame(q,idxs, boxes, confidences, classIDs,frame_out)
        success_out, frame_out = cameraCapture_out.read()

    cv2.destroyWindow('frame_in')
    cv2.destroyWindow('frame_out')
    cameraCapture_in.release()
    cameraCapture_out.release()