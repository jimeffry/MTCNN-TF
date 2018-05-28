#author: lxy
#time : 2018.3.19/12:40
#tool: python3
#version: 0.1
#modify:
################################
import cv2
import numpy as np
import sys
import os
import argparse

def args():
    parser = argparse.ArgumentParser(description='read video form file')
    parser.add_argument('--f_path',type=str,default="/home/lxy/Downloads/DataSet/bbb.mp4",\
                        help="input file dir")
    parser.add_argument('--saved_dir',type=str,default="/home/lxy/Downloads/DataSet/pedestrian_data/v5",\
                        help="saved dir")
    parser.add_argument('--num',type=int,default=15,\
                        help="every num frame to save image")
    return parser.parse_args()

def read_fil(path,out_dir,num):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("read video failed")
    print("have frame num: ",cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow("frame")
    total_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_cnt = 0
    cnt_fg = 0
    total_cnt =0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    while cap.isOpened():
        _,frame = cap.read()
        #total_cnt+=1
        #print("total: ",total_cnt,cnt_fg)
        #cv2.imshow("frame",frame)
        #cv2.waitKey(10)
        if cnt_fg == num:
            print("frame",frame_cnt)
            cv2.imwrite(out_dir+"/frame_{}.jpg".format(frame_cnt),frame)
            frame_cnt +=1
            cnt_fg = 0
        cnt_fg+=1
        if frame_cnt == int(total_f/num):
            break
    cap.release()
    cv2.destroyWindow("frame")

def remove_f(path):
    cnt_img=0
    if not os.path.exists(path):
        print("path is not exist")
    else:
        for one_file in os.listdir(path):
            img_path = os.path.join(path,one_file)
            if not os.path.isfile(img_path):
                print("img is not exist")
            else:
                img = cv2.imread(img_path)
                if img is None:
                    os.remove(img_path)
                else:
                    cnt_img +=1
                    continue
    print("img: ",cnt_img)

if __name__=='__main__':
    parm = args()
    f_p = parm.f_path
    o_dir = parm.saved_dir
    infer_num = parm.num
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    read_fil(f_p,o_dir,infer_num)
    #remove_f(o_dir)