'''
author: lxy
date-time: 2018/01/26 10:00
tool: python2.7
project: face_dect
References:
  1. Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao , " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter
  2. [MTCNN-MXNET](https://github.com/Seanlinx/mtcnn)
  3. [MTCNN-CAFFE](https://github.com/CongWeilin/mtcnn-caffe)
  4. [deep-landmark](https://github.com/luoyetx/deep-landmark)
vertion: v0.1
modify:
'''
import numpy as np
import sys
sys.path.append('..')
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
from train_models.mtcnn_model import P_Net,R_Net,O_Net
import cv2
import argparse
from train_models.MTCNN_config import config

test_relu =config.train_face

def parameter():
    parser = argparse.ArgumentParser(description='Mtcnn camera test')
    parser.add_argument("--min_size",type=int,default=24,\
                        help='determin the image pyramid and the lest is 12')
    parser.add_argument("--threshold",type=float,default=[0.7,0.99,0.8],nargs="+",\
                        help='filter the proposals according to score')
    parser.add_argument("--slid_window",type=bool,default=False,\
                        help='if true Pnet will use slid_window to produce proposals')
    parser.add_argument('--batch_size',type=int,default=[1,256,128],nargs="+",\
                        help='determin the pnet rnet onet input batch_size')
    parser.add_argument('--epoch_load',type=int,default=[10,2300,60],nargs="+",\
                        help='load the saved paramters for pnet rnet onet')
    parser.add_argument('--file_in',type=str,default='None',\
                        help='input file')
    return parser.parse_args()

def load_model(epoch_load):
    if test_relu==5 or test_relu==100:
        if config.rnet_wide:
            #5,500,60;  5,1700,60; 10,800,60; 10,2300,60; 1700,2300,60
            prefix = ["../data/MTCNN_model/PNet_landmark/PNet", "../data/MTCNN_model/RNet_landmark/rnet_wide/RNet", "../data/MTCNN_model/ONet_landmark/ONet"]
        else:
            # 5,40,60; 5,1200,60; 10,1400,60; 10,5200,60
            prefix = ["../data/MTCNN_model/PNet_landmark/PNet", "../data/MTCNN_model/RNet_landmark/RNet", "../data/MTCNN_model/ONet_landmark/ONet"]
    else:
        # 32,500,25
        if config.rnet_wide:
            prefix = ["../data/MTCNN_model/PNet_landmark/PNet", "../data/MTCNN_model/RNet_landmark/rnet_wide/RNet", "../data/MTCNN_model/ONet_landmark/v1_trained/ONet"]
        else:
            #epoch_load = [32,30,25],[32,4400,25]
            prefix = ["../data/MTCNN_model/PNet_landmark/v1_trained/PNet", "../data/MTCNN_model/RNet_landmark/v1_trained/RNet", "../data/MTCNN_model/ONet_landmark/v1_trained/ONet"]
            #[205,500,200]
            #prefix = ["../data/MTCNN_bright_model/PNet_landmark/PNet", "../data/MTCNN_bright_model/RNet_landmark/RNet", "../data/MTCNN_bright_model/ONet_landmark/ONet"]
            #pedestrain [80,360,200],[580,4900,600],[1600,4500,600],[1600,2900,4000]
            #prefix = ["../data/MTCNN_caltech_model/PNet_landmark/PNet", "../data/MTCNN_caltech_model/RNet_landmark/RNet", "../data/MTCNN_caltech_model/ONet_landmark/ONet"]
            #person voc[1600,2900,300]
            #prefix = ["../data/MTCNN_voc_model/PNet_landmark/PNet", "../data/MTCNN_voc_model/RNet_landmark/RNet", "../data/MTCNN_voc_model/ONet_landmark/ONet"]
    print("demo epoch load ",epoch_load)
    model_path = ["%s-%s" %(x,y ) for x, y in zip(prefix,epoch_load)]
    print("demo model path ",model_path)
    return model_path

def process_img():
    param = parameter()
    min_size = param.min_size
    score_threshold = param.threshold
    slid_window = param.slid_window
    if test_relu==100:
        batch_size = [1,1,1]
    else:
        batch_size = param.batch_size
    epoch_load = param.epoch_load
    multi_detector = [None,None,None]
    #load paramter path
    model_path = load_model(epoch_load)
    #load net result
    if slid_window:
        print("using slid window")
        Pnet_det = None
        return [None,None,None]
    else:
        Pnet_det = FcnDetector(P_Net,model_path[0])
    Rnet_det = Detector(R_Net,data_size=24,batch_size=batch_size[1],model_path=model_path[1])
    Onet_det = Detector(O_Net,data_size=48,batch_size=batch_size[2],model_path=model_path[2])
    multi_detector = [Pnet_det,Rnet_det,Onet_det]
    #get bbox and landmark
    Mtcnn_detector = MtcnnDetector(multi_detector,min_size,threshold=score_threshold)
    #bboxs,bbox_clib,landmarks = Mtcnn_detector.detect(img)
    return Mtcnn_detector

def add_label(img,bbox,landmark):
    num = bbox.shape[0]
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_scale =1
    thickness = 1
    for i in range(num):
        x1,y1,x2,y2 = int(bbox[i,0]),int(bbox[i,1]),int(bbox[i,2]),int(bbox[i,3])
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
        score_label = str('{:.2f}'.format(bbox[i,4]))
        size = cv2.getTextSize(score_label, font, font_scale, thickness)[0]
        if y1-int(size[1]) <= 0:
            cv2.rectangle(img, (x1, y2), (x1 + int(size[0]), y2+int(size[1])), (255, 0, 0))
            cv2.putText(img, score_label, (x1,y2+size[1]), font, font_scale, (255, 255, 255), thickness)
        else:
            cv2.rectangle(img, (x1, y1-int(size[1])), (x1 + int(size[0]), y1), (255, 0, 0))
            cv2.putText(img, score_label, (x1,y1), font, font_scale, (255, 255, 255), thickness)
    if landmark is not None:
        for i in range(landmark.shape[0]):
            for j in range(5):
                #print(int(landmark[i][2*j]),int(landmark[i][2*j+1]))
                cv2.circle(img, (int(landmark[i][2*j]),int(landmark[i][2*j+1])), 2, (0,0,255))

def camera(file_in):
    cv2.namedWindow("result")
    cv2.moveWindow("result",1400,10)
    #camera_cap = cv2.VideoCapture('/home/lxy/Develop/Center_Loss/face_detect/videos/profile_video.wmv')
    if file_in =='None':
        camera_cap = cv2.VideoCapture(0)
    else:
        camera_cap = cv2.VideoCapture(file_in)
    if not camera_cap.isOpened():
        print("failded open camera")
        return -1
    mtcnn_dec = process_img()
    while camera_cap.isOpened():
        ret,frame = camera_cap.read()
        if ret:
            bbox_clib,landmarks = mtcnn_dec.detect(frame)
            #print("landmark ",landmarks)
            if len(bbox_clib):
                add_label(frame,bbox_clib,landmarks)
            if (cv2.waitKey(1)& (0xFF == ord('q'))):
                break
            cv2.imshow("result",frame)
        else:
            print("can not find device")
            break
    camera_cap.release()
    cv2.destroyAllWindows()

def demo_img(file_in):
    cv2.namedWindow("result")
    cv2.moveWindow("result",1400,10)
    if file_in =='None':
        cv2.destroyAllWindows()
        print("please input right path")
        return -1
    else:
        img = cv2.imread(file_in)
    mtcnn_dec = process_img()
    bbox_clib,landmarks = mtcnn_dec.detect(img)
    if len(bbox_clib):
        add_label(img,bbox_clib,landmarks)
        cv2.imshow("result",img)
        cv2.waitKey(0)

if __name__ == '__main__':
    #process_img()
    arg = parameter()
    file_in = arg.file_in
    camera(file_in)
    #demo_img(file_in)
