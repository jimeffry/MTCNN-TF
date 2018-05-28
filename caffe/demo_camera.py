'''
author: lxy
date-time: 2018/04/8 16:00
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
from test import MtcnnDetector
import cv2
import argparse

test_relu =0

def parameter():
    parser = argparse.ArgumentParser(description='Mtcnn camera test')
    parser.add_argument("--min_size",type=int,default=24,\
                        help='determin the image pyramid and the lest is 12')
    parser.add_argument("--threshold",type=float,default=[0.6,0.7,0.9],nargs="+",\
                        help='filter the proposals according to score')
    parser.add_argument("--nms_threshold",type=float,default=[0.5,0.8,0.6],nargs="+",\
                        help='filter the proposals according to score')
    parser.add_argument('--batch_size',type=int,default=[1,256,32],nargs="+",\
                        help='determin the pnet rnet onet input batch_size')
    parser.add_argument('--epoch_load',type=int,default=[32,700,25],nargs="+",\
                        help='load the saved paramters for pnet rnet onet')
    parser.add_argument('--file_in',type=str,default='None',\
                        help='input file')
    return parser.parse_args()

def load_model(epoch_load):
    if test_relu==1:
        # 5,40,60
        prefix = ["../data/MTCNN_model/PNet_landmark/PNet", "../data/MTCNN_model/RNet_landmark/RNet", "../data/MTCNN_model/ONet_landmark/ONet"]
    else:
        #epoch_load = [32,30,25]
        prefix = ["../data/MTCNN_model/PNet_landmark/v1_trained/PNet", "../data/MTCNN_model/RNet_landmark/v1_trained/RNet", "../data/MTCNN_model/ONet_landmark/v1_trained/ONet"]
        #[205,500,200]
        #prefix = ["../data/MTCNN_bright_model/PNet_landmark/PNet", "../data/MTCNN_bright_model/RNet_landmark/RNet", "../data/MTCNN_bright_model/ONet_landmark/ONet"]
        #pedestrain [80,360,200],[580,4900,600],[1600,4500,600],[1600,2900,4000]
        #prefix = ["../data/MTCNN_caltech_model/PNet_landmark/PNet", "../data/MTCNN_caltech_model/RNet_landmark/RNet", "../data/MTCNN_caltech_model/ONet_landmark/ONet"]
    print("demo epoch load ",epoch_load)
    model_path = ["%s-%s" %(x,y ) for x, y in zip(prefix,epoch_load)]
    print("demo model path ",model_path)
    return model_path

def detector():
    param = parameter()
    min_size = param.min_size
    score_threshold = param.threshold
    nms_threshold = param.nms_threshold
    #get bbox and landmark
    Mtcnn_detector = MtcnnDetector(min_size,score_threshold,nms_threshold)
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
    mtcnn_dec = detector()
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
    #tmp = img[:, :, 2].copy()
    #img[:, :, 2] = img[:, :, 0]
    #img[:, :, 0] = tmp
    mtcnn_dec = detector()
    bbox_clib,landmarks = mtcnn_dec.detect(img)
    if len(bbox_clib):
        add_label(img,bbox_clib,landmarks)
        cv2.imshow("result",img)
        cv2.waitKey(0)

if __name__ == '__main__':
    #process_img()
    arg = parameter()
    file_in = arg.file_in
    #camera(file_in)
    demo_img(file_in)
