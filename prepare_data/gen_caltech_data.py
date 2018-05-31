#coding:utf-8
import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU,IoU_self
import argparse


def args():
    parser = argparse.ArgumentParser(description="generate Pnet train data")
    parser.add_argument('--anno_file',type=str,default="wider_face_train.txt",\
                        help='dataset annotation file')
    parser.add_argument('--img_dir',type=str,default="WIDER_train/images",\
                        help='images saved')
    parser.add_argument('--pos_save_dir',type=str,default="12/positive",\
                        help='positive images saved')
    parser.add_argument('--part_save_dir',type=str,default="12/part",\
                        help='part images saved')
    parser.add_argument('--neg_save_dir',type=str,default="12/negative",\
                        help='negtive images saved')
    parser.add_argument('--save_dir',type=str,default="./12",\
                        help='new annotions saved dir')
    parser.add_argument('--pos_txt',type=str,default="pos_12.txt",\
                        help='positive images annotion file ')
    parser.add_argument('--neg_txt',type=str,default="neg_12.txt",\
                        help='negtive images annotion file ')
    parser.add_argument('--part_txt',type=str,default="part_12.txt",\
                        help='part images annotion file ')
    parser.add_argument('--img_size',type=int,default=12,\
                        help='saved image size  ')
    return parser.parse_args()

def rd_anotation(annotation,im_dir):
    #annotation = annotation.strip().split('\t')
    annotation = annotation.strip().split()
    #image path
    #print("annotation split: ",annotation)
    im_path = annotation[0]
    #boxed change to float type
    bbox = map(float, annotation[1:])
    #gt
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    #load image
    #img = cv2.imread(os.path.join(im_dir, im_path + '.jpg')) 
    print(os.path.join(im_dir, im_path)) 
    img = cv2.imread(os.path.join(im_dir, im_path))          
    height, width, channel = img.shape
    return img,boxes,height,width

def gen_neg_data(img,boxes_gd,height,width,neg_save_dir,ng_id,f2,neg_hold=50,img_size=12,neg_iou=0.3):
    neg_num = 0
    #1---->50
    n_idx = ng_id
    while neg_num < neg_hold:
        #neg_num's size [40,min(width, height) / 2],min_size:40 
        size_h = npr.randint(40, min(width, height) / 2)
        size_w = np.int(size_h/2)
        #top_left
        nx = npr.randint(0, width - size_w)
        ny = npr.randint(0, height - size_h)
        #random crop
        crop_box = np.array([nx, ny, nx + size_w, ny + size_h])
        #cal iou
        Iou = IoU(crop_box, boxes_gd)
        #Iou = IoU_self(crop_box, boxes_gd)
            
        cropped_im = img[ny : ny + size_h, nx : nx + size_w, :]
        resized_im = cv2.resize(cropped_im, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < neg_iou:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            #f2.write("12/negative/%s.jpg"%n_idx + ' 0\n')
            f2.write(save_file + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1
    return n_idx

def gen_pneg_data(img,boxes_gd,top_point,neg_save_dir,ng_id,f2,pneg_hold=5,img_size=12,neg_iou=0.3):
    n_idx = ng_id
    x1,y1,h,w,height,width,_,_ = top_point
    for i in range(pneg_hold):
        #print("h, ",h,width,height)
        #size_h = npr.randint(int(h), min(width, height) / 2)
        size_h = npr.randint(int(h/2), int(h))
        size_w = npr.randint(int(w/2), int(w))
        #size_w = size_h/2
        # delta_x and delta_y are offsets of (x1, y1)
        delta_x = npr.randint(max(-size_w, -x1), w)
        delta_y = npr.randint(max(-size_h, -y1), h)
        nx1 = int(max(0, x1 + delta_x))
        ny1 = int(max(0, y1 + delta_y))
        if nx1 + size_w > width or ny1 + size_h > height:
            continue
        crop_box = np.array([nx1, ny1, nx1 + size_w, ny1 + size_h])
        #Iou = IoU(crop_box, boxes_gd)
        Iou = IoU_self(crop_box, boxes_gd)
        cropped_im = img[ny1: ny1 + size_h, nx1: nx1 + size_w, :]
        resized_im = cv2.resize(cropped_im, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        if np.max(Iou) < neg_iou:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            f2.write(save_file + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
    return n_idx 

def gen_pos_part_data(img,boxes_gd,top_point,pos_save_dir,part_save_dir,p_id,d_id,f1,f3,pos_hold=20,img_size=12,piou=0.65,paiou=0.4):
    x1,y1,h,w,height,width,x2,y2 = top_point
    p_idx = p_id
    d_idx = d_id
    for i in range(pos_hold):
        # pos and part face size [minsize*0.8,maxsize*1.25]
        #print("bbox ",x1,y1,w,h)
        #size_h = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
        size_h = npr.randint(int(h), np.ceil(1.2 *h))
        #size_h = int(h)
        #size_w = int(w) 
        #print("h ",size_h)
        size_w = npr.randint(int(w), np.ceil(1.2 *w))
        #size_w = size_h/2
        # delta here is the offset of box center
        delta_x = npr.randint(-w * 0.2, w * 0.2)
        #delta_y = npr.randint(-h * 0.2, h * 0.2)
        delta_y = npr.randint(-h * 0.2, h * 0.2)
        #show this way: nx1 = max(x1+w/2-size/2+delta_x)
        nx1 = int(max(x1 + w / 2 + delta_x - size_w / 2, 0))
        #show this way: ny1 = max(y1+h/2-size/2+delta_y)
        ny1 = int(max(y1 + h / 2 + delta_y - size_h / 2, 0))
        nx2 = nx1 + size_w
        ny2 = ny1 + size_h
        if nx2 > width or ny2 > height:
            continue 
        crop_box = np.array([nx1, ny1, nx2, ny2])
        #yu gt de offset
        offset_x1 = (x1 - nx1) / float(size_w)
        offset_y1 = (y1 - ny1) / float(size_h)
        offset_x2 = (x2 - nx2) / float(size_w)
        offset_y2 = (y2 - ny2) / float(size_h)
        #crop
        cropped_im = img[ny1 : ny2, nx1 : nx2, :]
        #resize
        resized_im = cv2.resize(cropped_im, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        box_ = boxes_gd.reshape(1, -1)
        #print(crop_box,box_)
        #print("IOU",IoU(crop_box, box_))
        #if IoU(crop_box, box_) >= piou and delta_y<=0:
        if IoU(crop_box, box_) >= piou :
            save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
            f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
            cv2.imwrite(save_file, resized_im)
            p_idx += 1
        #elif IoU(crop_box, box_) <= paiou and delta_y>0:
        elif IoU(crop_box, box_) <= paiou and IoU(crop_box, box_) >0.2:
            save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
            f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
            cv2.imwrite(save_file, resized_im)
            d_idx += 1
    return p_idx,d_idx


def gen_pnet_data(img_num,Iou_thresh):
    neg_hold,pneg_hold,pos_hold = img_num
    neg_iou,piou,paiou = Iou_thresh
    parm = args()
    anno_file = parm.anno_file
    im_dir = parm.img_dir
    pos_save_dir = parm.pos_save_dir
    part_save_dir = parm.part_save_dir
    neg_save_dir = parm.neg_save_dir
    save_dir = parm.save_dir
    img_size = parm.img_size
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
    f1 = open(os.path.join(save_dir, parm.pos_txt), 'w')
    f2 = open(os.path.join(save_dir, parm.neg_txt), 'w')
    f3 = open(os.path.join(save_dir, parm.part_txt), 'w')
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print("%d pics in total" % num)
    p_idx = 0 # positive
    n_idx = 0 # negative
    d_idx = 0 # dont care
    idx = 0
    box_idx = 0
    pass_cnt =0
    cnt_pos_neg =0
    for annotation in annotations:
        img,boxes_gd,height,width = rd_anotation(annotation,im_dir)
        idx += 1
        if idx % 100 == 0:
            print(idx, "images done")
        ng_id = gen_neg_data(img,boxes_gd,height,width,neg_save_dir,n_idx,f2,neg_hold=neg_hold,neg_iou=neg_iou,img_size=img_size)
        n_idx = ng_id
        #as for 正 part样本
        for box in boxes_gd:
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            #gt's width
            w = x2 - x1 + 1
            #gt's height
            h = y2 - y1 + 1

            # ignore small faces
            # in case the ground truth boxes of small faces are not accurate
            if min(w, h) < 25 or x1 < 0 or y1 < 0:
                pass_cnt+=1
                continue
            top_point = [x1,y1,h,w,height,width,x2,y2]
            pneg_id = gen_pneg_data(img,boxes_gd,top_point,neg_save_dir,n_idx,f2,pneg_hold=pneg_hold,neg_iou=neg_iou,img_size=img_size)
            cnt_pos_neg +=(pneg_id-n_idx)
            n_idx = pneg_id       
        # generate positive examples and part faces
            pos_id,par_id = gen_pos_part_data(img,box,top_point,pos_save_dir,part_save_dir,p_idx,d_idx,f1,f3,pos_hold=pos_hold,piou=piou,paiou=paiou,img_size=img_size)
            p_idx,d_idx = [pos_id,par_id]
            box_idx += 1
        print "%s images done, pos: %s part: %s neg: %s, pass: %s, pn: %s"%(idx, p_idx, d_idx, n_idx,pass_cnt,cnt_pos_neg)
    f1.close()
    f2.close()
    f3.close()

if __name__=='__main__':
    #neg_hold,pneg_hold,pos_hold
    img_num = [50,5,20]
    #neg_iou,piou,paiou
    Iou_thresh = [0.2,0.7,0.4]
    gen_pnet_data(img_num,Iou_thresh)