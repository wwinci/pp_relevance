import sqlite3

import numpy
import cv2
import os
import time

yolo_dir = 'model_4classes'  # YOLO文件路径
weightsPath = os.path.join(yolo_dir, 'yolov3-voc_2000.weights')  # 权重文件
configPath = os.path.join(yolo_dir, 'yolov3-voc.cfg')  # 配置文件
labelsPath = os.path.join(yolo_dir, 'coco.names')  # label名称windows darknet python
imgPath = os.path.join(yolo_dir, 'test.jpg')  # 测试图像
CONFIDENCE = 0.5  # 过滤弱检测的最小概率
THRESHOLD = 0.3  # 非最大值抑制阈值

database = './FaceBase.db'
datasets = './datasets'
userInfo = {'stu_id': ''}

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


def detect(success,frame):
    faceRecordCount = 0
    flag = 0
    while success and cv2.waitKey(1) == -1 and not clicked:  # 当循环没结束，并且剩余的帧数大于零时进行下面的程序
        blobImg = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (416, 416), None, True,
                                        False)  # # net需要的输入是blob格式的，用blobFromImage这个函数来转格式
        net.setInput(blobImg)

        # 获取网络输出层信息（所有输出层的名字），设定并前向传播
        outInfo = net.getUnconnectedOutLayersNames()  # # 前面的yolov3架构也讲了，yolo在每个scale都有输出，outInfo是每个scale的名字信息，供net.forward使用
        start = time.time()
        layerOutputs = net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息，是二维结构。
        end = time.time()
        #print("[INFO] YOLO took {:.6f} seconds".format(end - start))  # # 可以打印下信息

        # 拿到图片尺寸
        (H, W) = frame.shape[:2]
        # 过滤layerOutputs
        # layerOutputs的第1维的元素内容: [center_x, center_y, width, height, objectness, N-class score data]
        # 过滤后的结果放入：
        boxes = []  # 所有边界框（各层结果放一起）
        confidences = []  # 所有置信度
        classIDs = []  # 所有分类ID

        # # 1）过滤掉置信度低的框框
        for out in layerOutputs:  # 各个输出层
            for detection in out:  # 各个框框
                # 拿到置信度
                scores = detection[5:]  # 各个类别的置信度
                classID = numpy.argmax(scores)  # 最高置信度的id即为分类id
                confidence = scores[classID]  # 拿到置信度

                # 根据置信度筛查
                if confidence > CONFIDENCE:
                    box = detection[0:4] * numpy.array([W, H, W, H])  # 将边界框放会图片尺寸
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)


        # # 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)  # boxes中，保留的box的索引index存入idxs
        # 得到labels列表
        with open(labelsPath, 'rt') as f:
            labels = f.read().rstrip('\n').split('\n')
        # 应用检测结果
        numpy.random.seed(42)
        COLORS = numpy.random.randint(0, 255, size=(len(labels), 3),
                                      dtype="uint8")  # 框框显示颜色，每一类有不同的颜色，每种颜色都是由RGB三个值组成的，所以size为(len(labels), 3)
        if len(idxs) > 0:
            for i in idxs.flatten():
                count = 0
                (w, h) = (boxes[i][2], boxes[i][3])
                if w*h > 6000 and classIDs[i] == 3:
                    count += 1

            if count > 0 and flag == 0:
                stu_id = addUserID()
                initDb()
                migrateToDb(stu_id)
                flag = 1

            elif count > 0 and flag == 1:
                pass

            elif count == 0 and flag == 1:
                flag = 0
                faceRecordCount = 0

            elif count == 0 and flag == 0:
                pass

            for i in idxs.flatten():  # indxs是二维的，第0维是输出层，所以这里把它展平成1维

                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                if classIDs[i] == 3:
                    #print(w*h)
                    if w * h > 6000:
                        FaceRecord(i,frame,boxes,stu_id,faceRecordCount)
                        faceRecordCount += 1
                else:
                    pass

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # 线条粗细为2px
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                            2)  # cv.FONT_HERSHEY_SIMPLEX字体风格、0.5字体大小、粗细2px
        cv2.imshow('detected image', frame)
        success, frame = cameraCapture.read()  # 摄像头获取下一帧

def FaceRecord(i,frame,boxes,stu_id,faceRecordCount):
    (x, y) = (boxes[i][0], boxes[i][1])
    (w, h) = (boxes[i][2], boxes[i][3])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not os.path.exists('{}/stu_{}'.format(datasets, stu_id)):
        os.makedirs('{}/stu_{}'.format(datasets, stu_id))

    cv2.imwrite('{}/stu_{}/img.{}.jpg'.format(datasets, stu_id, faceRecordCount + 1),
                gray[y - 10:y + h + 10, x - 10:x + w + 10])





def addUserID():
    dir = os.listdir('E:\desk/pp_relevance/datasets')
    if dir == []:
        new_id = '10000'
    else:
        dir_num = len(dir)
        lastdir = dir[dir_num - 1]
        new_id = str(int(lastdir[4:16]) + 1)

    return new_id

# 初始化数据库
def initDb():

    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    try:
        # 检测人脸数据目录是否存在，不存在则创建
        if not os.path.isdir(datasets):
            os.makedirs(datasets)

        # 查询数据表是否存在，不存在则创建
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                              stu_id VARCHAR(5) PRIMARY KEY NOT NULL,
                              face_id INTEGER DEFAULT -1,
                              created_time DATE DEFAULT (date('now','localtime'))
                              )
                          ''')
        # 查询数据表记录数
        cursor.execute('SELECT Count(*) FROM users')
        result = cursor.fetchone()
        dbUserCount = result[0]
        print("原用户数：",dbUserCount)
    except Exception as e:
        print('错误类型是', e.__class__.__name__)
        print('错误明细是', e)
        print("初始化数据库失败")

def migrateToDb(stu_id):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (stu_id) VALUES (?)',
                   (stu_id,))
        cursor.execute('SELECT Count(*) FROM users')

        result = cursor.fetchone()
        dbUserCount = result[0]
        print("插入新用户成功！")
        print("现用户数：",dbUserCount)


    except Exception as e:
        print('错误类型是', e.__class__.__name__)
        print('错误明细是', e)
        print("插入失败")

    finally:
        cursor.close()
        conn.commit()
        conn.close()



if __name__ == '__main__':
    cameraCapture = cv2.VideoCapture(0)  # 打开编号为0或1的摄像头
    cameraCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cameraCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow('detected image')  # 给视频框命名
    cv2.setMouseCallback('detected image', onMouse)
    print('显示摄像头图像，点击鼠标左键或按任意键退出')
    success, frame = cameraCapture.read()

    detect(success, frame)

    cv2.destroyWindow('detected image')
    cameraCapture.release()

