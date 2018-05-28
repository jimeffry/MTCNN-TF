#author: lxy
#time: 14:30 2018.3.29
#tool: python3
#version: 0.1
#project: pedestrian detection
#modify:
#####################################################
import numpy as np 
import os
import xml.etree.cElementTree as ET
import string 
import cv2

def read_xml_gtbox_and_label(xml_path,img_name):

    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
           and has [xmin, ymin, xmax, ymax, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    cnt_pass = 0
    cnt_total =0
    #print("process image ",img_name)
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'
        if child_of_root.tag == 'filename':
            assert child_of_root.text == img_name, 'image name and xml is not the same'
        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name' and not(child_item.text == "person"):
                    #label = NAME_LABEL_MAP[child_item.text]
                    print("is not person: ",xml_path)
                    break
                if child_item.tag == 'truncated' and int(child_item.text)==1:
                    cnt_pass +=1
                    return img_height,img_width,None,cnt_pass,0
                if child_item.tag == 'bndbox':
                    #tmp_box = []
                    for node in child_item:
                        if node.tag == 'xmin':
                            x1 = np.int32(float(node.text))
                        if node.tag == 'ymin':
                            y1 = np.int32(float(node.text))
                        if node.tag == 'xmax':
                            x2= np.int32(float(node.text))
                        if node.tag == 'ymax':
                            y2 = np.int32(float(node.text))
                    #tmp_box.append([x1,y1,x2,y2])  # [x1, y1. x2, y2]
                    #assert label is not None, 'label is none, error'
                    #tmp_box.append(label)  # [x1, y1. x2, y2, label]
                    box_list.append([x1,y1,x2,y2])
                    cnt_total+=1
    if len(box_list)>0:
        gtbox_label = np.array(box_list, dtype=np.int32)  # [x1, y1. x2, y2]
    else:
        gtbox_label = None
    return img_height, img_width, gtbox_label,cnt_pass,cnt_total

def transform_(base_dir,file_in,file_out):
    annotation_f = open(file_in,'r')
    f_out = open(file_out,'w')
    p_cnt = 0
    cnt_none = 0
    cnt_total =0
    for one_line in annotation_f.readlines():
        annot_splits = one_line.strip().split()
        img_name = annot_splits[0].split('/')[-1]
        xml_path = annot_splits[1]
        xml_path = os.path.join(base_dir,xml_path)
        h,w, gtboxes,cnt_pass,cnt = read_xml_gtbox_and_label(xml_path,img_name)
        p_cnt+=cnt_pass
        cnt_total+=cnt
        if gtboxes is  None:
            print("gbox is none: ",annot_splits[0])
            cnt_none+=1
            continue
        else:
            
            img = cv2.imread(os.path.join(base_dir,annot_splits[0]))
            for box in gtboxes:
                cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0))
            cv2.imshow("imgshow",img)
            cv2.waitKey(1000)
            '''
            f_out.write("{} ".format(annot_splits[0]))
            row,col = np.shape(gtboxes)
            for i in range(row):
                for j in range(col):
                    f_out.write("{} ".format(gtboxes[i][j]))
            f_out.write("\n")
            '''
    annotation_f.close()
    f_out.close()
    print("pass: %s, none: %s, person: %s" %(p_cnt,cnt_none,cnt_total))

if __name__=='__main__':
    base_dir = "/home/lxy/Downloads/DataSet/VOC_Person"
    file_in = "/home/lxy/Downloads/DataSet/VOC_Person/train.txt"
    file_out = "train_voc.txt"
    transform_(base_dir,file_in,file_out)
