import sqlite3
import numpy as np
import cv2
import os
import time
from detector import initDb

yolo_dir = 'model_4classes'  # YOLO文件路径
weightsPath = os.path.join(yolo_dir, 'yolov3-voc_2000.weights')  # 权重文件
configPath = os.path.join(yolo_dir, 'yolov3-voc.cfg')  # 配置文件
labelsPath = os.path.join(yolo_dir, 'coco.names')  # label名称windows darknet python
imgPath = os.path.join(yolo_dir, 'test.jpg')  # 测试图像
CONFIDENCE = 0.5  # 过滤弱检测的最小概率
THRESHOLD = 0.4  # 非最大值抑制阈值

database = './FaceBase.db'
datasets = './datasets'
userInfo = {'stu_id': ''}


def train():
    try:
        if not os.path.isdir(datasets):
            raise FileNotFoundError

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.exists('./recognizer'):
            os.makedirs('./recognizer')
        faces, labels = prepareTrainingData(datasets)
        face_recognizer.train(faces, np.array(labels))
        face_recognizer.save('./recognizer/trainingData.yml')
    except FileNotFoundError:
        print("未发现人脸数据")
    except Exception as e:
        print('错误类型是', e.__class__.__name__)
        print('错误明细是', e)

    else:
        print("人脸数据训练完成")
        initDb()



def prepareTrainingData(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []

    face_id = 1
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    # 遍历人脸库
    for dir_name in dirs:
        if not dir_name.startswith('stu_'):
            continue
        stu_id = dir_name.replace('stu_', '')
        try:
            cursor.execute('SELECT * FROM users WHERE stu_id=?', (stu_id,))
            ret = cursor.fetchall()
            cursor.execute('UPDATE users SET face_id=? WHERE stu_id=?', (face_id, stu_id,))

        except Exception as e:
            print('错误类型是', e.__class__.__name__)
            print('错误明细是', e)
            print("初始化数据库失败")

        subject_dir_path = data_folder_path + '/' + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith('.'):
                continue
            image_path = subject_dir_path + '/' + image_name
            image = cv2.imread(image_path)
            face = detectFace(image)

            if face is not None:
                faces.append(face)
                labels.append(face_id)
        face_id = face_id + 1

    cursor.close()
    conn.commit()
    conn.close()

    return faces, labels

def detectFace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray



if __name__ == '__main__':
    train()