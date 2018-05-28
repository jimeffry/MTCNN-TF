#coding:utf-8
import sys
#sys.path.append("../")
sys.path.insert(0,'..')
import numpy as np
import argparse
import os
import cPickle as pickle
import cv2
from train_models.mtcnn_model import P_Net,R_Net
from train_models.MTCNN_config import config
from loader import TestLoader
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
from utils import convert_to_square,IoU,convert_to_rect,IoU_self
from data_utils import get_path,read_annotation
import pdb
#net : 24(RNet)/48(ONet)
#data: dict()
'''
def args():
    parser = argparse.ArgumentParser(description="gen_hard_example for Rnet Onet")
    parser.add_argument('--net',type=str,required=True,default='RNet'
                    help='should be RNet of ONet')
    return parser.parse_args()
'''
def save_hard_example(gen_anno_file, gen_imgs_dir,data,save_path,test_mode):
    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image

    im_idx_list = data['images']
    # print(images[0])
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)
    print("processing %d images in total" % num_of_images)    
    # save files
    print("saved hard example dir ",net)
    #neg_label_file = "%s/neg_%d.txt" % (net, image_size)
    neg_label_file = gen_anno_file[0]
    neg_file = open(neg_label_file, 'w')
    pos_label_file = gen_anno_file[1]
    pos_file = open(pos_label_file, 'w')
    part_label_file = gen_anno_file[2]
    part_file = open(part_label_file, 'w')
    #read detect result
    det_boxes = pickle.load(open(os.path.join(save_path, 'detections.pkl'), 'rb'))
    print("det boxes and image num: ",len(det_boxes), num_of_images)
    #print len(det_boxes)
    #print num_of_images
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    cnt_pass =0
    #im_idx_list image index(list)
    #det_boxes detect result(list)
    #gt_boxes_list gt(list)
    neg_dir,pos_dir,part_dir = gen_imgs_dir
    if test_mode == "PNet" and not config.train_face:
        X1_thresh = 0.45
        Y1_thresh = -0.2
    elif test_mode == "RNet" and not config.train_face:
        Y1_thresh = -0.2
        X1_thresh = 0.45
        print("generate Onet")
    else:
        Y1_thresh = 1
        X1_thresh = 1
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        #change to square
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1
            # ignore box that is too small or beyond image border
            #if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
            if  x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1 or width <=10 :
                #print("pass")
                cnt_pass+=1
                continue
            # compute intersection over union(IoU) between current box and all gt boxes
            Iou_ = IoU(box, gts)
            Iou_gt = IoU_self(box,gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)
            # save negative images and write label
            # Iou with all gts must below 0.3   
            union_max = np.max(Iou_) 
            gt_max = np.max(Iou_gt)        
            if union_max <=0.3 and neg_num < 60:
                #save the examples
                idx = np.argmax(union_max)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)
                save_file = get_path(neg_dir, "%s.jpg" % n_idx)
                # print(save_file)
                #neg_file.write(save_file + ' 0\n')
                neg_file.write(save_file + ' 0  %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                '''
                if union_max>0:
                    if np.abs(offset_x1) < 1 :
                        neg_file.write(save_file + ' 0  %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                        #neg_file.write('   %.2f %.2f %.2f %.2f' % (x1, y1, x2, y2))
                        #neg_file.write('   %.2f %.2f %.2f %.2f  ' % (x_left, y_top, x_right, y_bottom))
                        #neg_file.write(im_idx +'\n')
                        cv2.imwrite(save_file, resized_im)
                        n_idx += 1
                else:
                    neg_file.write(save_file + ' 0  %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1
                '''
                neg_num += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou_)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if union_max >= 0.6:
                    #if np.max(Iou) >= 0.65:
                    #if union_max >= 0.7 and offset_y1>Y1_thresh and np.abs(offset_x1)<= X1_thresh:
                    save_file = get_path(pos_dir, "%s.jpg" % p_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                    #elif np.max(Iou) >= 0.3:
                elif union_max > 0.3 and union_max <=0.4:
                    #elif union_max <= 0.3 and union_max >0.1 and offset_y1 <Y1_thresh and np.abs(offset_x1)<= X1_thresh: 
                    save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
        print("%s images done, pos: %s part: %s neg: %s, pass: %s"%(image_done, p_idx, d_idx, n_idx,cnt_pass))
    neg_file.close()
    part_file.close()
    pos_file.close()
    print("neg image num: ",n_idx)
    print("pos image num: ",p_idx)
    print("pat image num: ",d_idx)
    print("pass num : ",cnt_pass)

def rd_anotation(img_saved_dir,filename,data_set_name):
    data = dict()
    image_path_list = []
    boxes_gd = []
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    for annotation in annotations:
        annotation = annotation.strip().split()
        im_path = annotation[0]
        if data_set_name == "WiderFace":
            im_path = im_path +'.jpg'
        im_path = os.path.join(img_saved_dir,im_path)
        #print("img path ",im_path)
        image_path_list.append(im_path)
        #boxed change to float type
        bbox = map(float, annotation[1:])
        #print("box : ",bbox)
        #gt
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        boxes_gd.append(boxes)
    data['images'] = image_path_list
    data['bboxes'] = boxes_gd
    return data

def t_net(prefix, epoch,batch_size, img_saved_dir,anno_file,gen_anno_file,gen_imgs_dir,data_set_name,ignore_det=False,test_mode="PNet",thresh=[0.6, 0.6, 0.7], min_face_size=25,\
             stride=2):
    slide_window=False
    detectors = [None, None, None]
    print("Test model: ", test_mode)
    #PNet-echo
    print("epoch num ",epoch[0])
    ''' #for Pnet test
    epoch_num = epoch[0]
    epoch_c = np.arange(2,epoch_num,2)
    prefix_c = []
    prefix = prefix[0]
    [prefix_c.append(prefix) for i in range(len(epoch_c))]
    '''
    print("prefixs is ",prefix)
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    #print("after zip model_path is ",model_path)
    #model_path[0] = prefix + '-'+str(epoch_num) #for Pnet test
    print("model_path 0 is ",model_path[0])
    # load pnet model
    if slide_window:
        PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["RNet", "ONet"]:
        print("==================================", test_mode)
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    # load onet model
    if test_mode == "ONet":
        print("==================================", test_mode)
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet  
    #read annatation(type:dict)
    #img_box_dic = read_annotation(img_saved_dir,anno_file)
    img_box_dic = rd_anotation(img_saved_dir,anno_file,data_set_name)
    print("gen_hardexample  threshold ",thresh)
    if not ignore_det:
        mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh)
    print("==================================")
    # 注意是在“test”模式下
    # imdb = IMDB("wider", image_set, root_path, dataset_path, 'test')
    # gt_imdb = imdb.gt_imdb()
    test_data = TestLoader(img_box_dic['images'])
    #list
    if not ignore_det:
        detections,_ = mtcnn_detector.detect_face(test_data)
    if test_mode == "PNet":
        save_net = "RNet"
        save_path = '24/RNet'
    elif test_mode == "RNet":
        save_net = "ONet"
        save_path = "48/ONet"
    #save detect result
    #save_path = os.path.join(data_dir, save_net)
    print ("save path is",save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path, "detections.pkl")
    if not ignore_det:
        with open(save_file, 'wb') as f:
            pickle.dump(detections, f,1)
        f.close()
    print("%s Test is Over and begin OHEM" % image_size)
    save_hard_example(gen_anno_file, gen_imgs_dir,img_box_dic, save_path,test_mode)


def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be PNet, RNet or ONet',
                        default='PNet', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=["../data/MTCNN_model/PNet_landmark/v1_trained/PNet", "../data/MTCNN_model/RNet_landmark/RNet", "../data/MTCNN_model/ONet_landmark/ONet"],
                        type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[32, 2900, 22], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[1, 2048, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.4, 0.6, 0.7], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=24, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=2, type=int)
    parser.add_argument('--anno_file',type=str,default="./wider_face_train.txt",\
                        help='annotation saved file path')
    parser.add_argument('--img_saved_dir',type=str,default="./WIDER_train/images/",\
                        help='images saved path')
    parser.add_argument('--pos_txt',type=str,default="pos24.txt",\
                        help='positive images annotion file ')
    parser.add_argument('--neg_txt',type=str,default="neg24.txt",\
                        help='negtive images annotion file ')
    parser.add_argument('--part_txt',type=str,default="part24.txt",\
                        help='part images annotion file ')
    parser.add_argument('--train_data_set',type=str,default="WiderFace",\
                        help='the model will be trained in the dataset ')  
    parser.add_argument('--ignore_det',type=bool,default=False,\
                        help='only run save_hard_example ')  
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    #net = 'RNet'
    img_saved_dir = args.img_saved_dir
    anno_file = args.anno_file
    neg_label_file = args.train_data_set+"_"+args.neg_txt
    pos_label_file = args.train_data_set+"_"+args.pos_txt
    part_label_file = args.train_data_set+"_"+args.part_txt
    prefix = args.prefix
    epoch_list = args.epoch
    batch_size = args.batch_size
    stride = args.stride
    test_mode = args.test_mode
    score_thresh = args.thresh
    min_face_size = args.min_face
    ignore_det = args.ignore_det
    if args.test_mode == "ONet":
        image_size = 48
    if args.test_mode =="PNet":
        net = "RNet"
    elif args.test_mode == "RNet":
        net = "ONet"
    if net == "RNet":
        image_size = 24
    if net == "ONet":
        image_size = 48
    data_dir = '%s' % str(image_size)
    neg_label_file = os.path.join(data_dir,neg_label_file)
    pos_label_file = os.path.join(data_dir,pos_label_file)
    part_label_file = os.path.join(data_dir,part_label_file)
    gen_anno_file = [neg_label_file,pos_label_file,part_label_file]
    data_set_name = args.train_data_set
    neg_dir = get_path(data_dir, '%s_negative' %(data_set_name))
    pos_dir = get_path(data_dir, '%s_positive' %(data_set_name))
    part_dir = get_path(data_dir, '%s_part' %(data_set_name))
    gen_imgs_dir = [neg_dir,pos_dir,part_dir]
    #create dictionary shuffle   
    for dir_path in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)    
    print ('Called with argument:')
    print("config ",config.train_face)
    t_net(prefix, epoch_list,batch_size, img_saved_dir,anno_file,gen_anno_file,gen_imgs_dir,data_set_name,ignore_det,test_mode,score_thresh, min_face_size,stride)

