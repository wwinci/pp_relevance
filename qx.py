import cv2
import torch
import numpy as np
from get_feature_reid import Cal_features
import detector
import recognizer
import os

class Person(object):
    def __init__(self, id=None):
        self.bags = []
        self.id = id

class Queue(object):

    def __init__(self, match_mod='L2', L2_match_distance=20000, iou_threshold=0.5, bcosim=0.92):

        self.personQ = []
        self.bagQ = []
        
        self.match_mod = match_mod
        self.L2_match_distance = L2_match_distance
        self.iou_threshold = iou_threshold
        self.bcosim = bcosim

        self.cal = Cal_features()

    def iou(self, person, bag):

        x1 = max(person[0], bag[0])
        x2 = min(person[1], bag[1])
        y1 = max(person[2], bag[2])
        y2 = min(person[3], bag[3])

        inter_area = max(x2-x1+1, 0) * max(y2-y1+1, 0)
        bag_area = (bag[1]-bag[0]+1) * (bag[3]-bag[2]+1)
        return inter_area / bag_area

    def match(self, person, bag):
        
        if self.match_mod == 'L2':
            pc = ((person[0]+person[1])/2, (person[2]+person[3])/2)
            bc = ((bag[0]+bag[1])/2, (bag[2]+bag[3])/2)
            distance = (pc[0]-bc[0])**2 + (pc[1]-bc[1])**2
            result = self.L2_match_distance - distance
        
        if self.match_mod == "IOU":
            distance = self.iou(person, bag)
            result = distance - self.iou_threshold

        return result # positive means AC & bigger is better

    def update(self, cam, personPos, personId, bagPos, bagFeat=None, frame=None):
        """ update the queue with bboxs and id
        
        Args:
            cam: 'in' or 'out'
            personPos(list/tensor): 2d, n * 4
            personId(list/tensor): 1d, n
            bagPos(list/tensor): 2d, m * 4
            bagFeat: None, or 2d, m * featSize
            frame(npArray): 3d, 3 * height * width
        """
        if len(personPos) == 0:
            return
        
        if type(personPos) == list:
            personPos = torch.tensor(personPos)
        if type(bagPos) == list:
            bagPos = torch.tensor(bagPos)
        if type(bagFeat) == list:
            bagFeat = torch.tensor(bagFeat)

        if bagFeat == None:
            bagFeat = []
            for x1,x2,y1,y2 in bagPos:
                bagFeat.append(self.cal.get_features(frame[:, x1:x2, y1:y2])) # TODO frame 格式?

        if (cam == 'in'):
            for pid in personId:
                if pid not in list(map(lambda x: x.id, self.personQ[-3:])): # 最近3个人
                    print('New person, person_id', pid)
                    self.personQ.append(Person(pid))
            for i, bf in enumerate(bagFeat):
                unique = True
                for bag in self.bagQ[-3:]: # 最近3个包
                    # print(torch.cosine_similarity(bf, bag).item())
                    if (torch.cosine_similarity(bf, bag).item() > self.bcosim):
                        unique = False
                        break
                if unique:                    
                    # 新包和现场人距离最近者匹配
                    self.bagQ.append(bf)
                    dis = list(map(lambda x: self.match(x, bagPos[i]), personPos))
                    maxid = np.array(dis).argmax()
                    pQid = list(map(lambda x: x.id, self.personQ)).index(personId[maxid])
                    
                    print('New bag, person_id {}'.format(self.personQ[pQid].id))
                    self.personQ[pQid].bags.append(bf)
                    
        if (cam == 'out'):

            for i, bf in enumerate(bagFeat):
                
                f = None
                for bag in self.bagQ[:3]: # 最早3个包
                    if (torch.cosine_similarity(bf, bag).item() > self.bcosim):
                        f = bag
                        break
                
                if f != None:

                    dis = list(map(lambda x: self.match(x, bagPos[i]), personPos))
                    maxid = np.array(dis).argmax()
                    idList = list(map(lambda x: x.id, self.personQ))
                    if personId[maxid] not in idList:
                        continue
                    pQid = idList.index(personId[maxid])

                    sim = list(map(lambda x: torch.cosine_similarity(bf, x).item(), self.personQ[pQid].bags))
                    bid = np.array(sim).argmax()
                    if (sim[bid] > self.bcosim):
                        print('Bag matched, person_id: {}'.format(personId[maxid]))
                        self.personQ[pQid].bags.pop(bid)
                        self.bagQ.pop(self.bagQ.index(f))

                    else:
                        print('Bag not matched, person_id: {}'.format(personId[maxid]))
                        self.personQ[pQid].bags.pop(bid)
                        self.bagQ.pop(self.bagQ.index(f))
                
                else:
                    print('Unknown bag')

            while len(self.personQ)>0 and len(self.personQ[0].bags)==0:
                print('Person out, person_id: {}'.format(self.personQ[0].id))
                self.personQ.pop(0)

def imgReading(frame):

    img = torch.from_numpy(frame).float() / 255.0
    img = img.permute(2, 0, 1).contiguous()
    return img






