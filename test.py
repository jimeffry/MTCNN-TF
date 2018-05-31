    # -*- coding: utf-8 -*-
###############################################
#created by :  lxy 
#Time:  2017/07/12 14:09
#project: Face recognize
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified: 2018.5.31
####################################################
import xml.etree.cElementTree as ET
import sys
import cv2
import shutil
'''
tree = ET.parse("/home/lxy/Downloads/DataSet/VOC_Person/VOC2012/Annotations/2007_000664.xml")
root = tree.getroot()
for child_of_root in root:
    if child_of_root.tag == 'object':
        for child_item in child_of_root:
            print(child_item.tag) 
'''
def cp_img():
    img_path=sys.argv[1]
    img_file=open(img_path,'r')
    count=0
    lines_ = img_file.readlines()
    for line in lines_ :
        count = count+1
        if count >49 and count <4000:
            tem_str=line.strip().split()
            file1="/home/lxy/Develop/Center_Loss/MTCNN-Tensorflow/prepare_data/"+tem_str[0]
            shutil.copyfile(file1,"/home/lxy/Develop/Center_Loss/MTCNN-Tensorflow/prepare_data/48/market_part/"+str(count)+".jpg")
            #img_name="img"+str(count)
            #shutil.copyfile(line.strip(),"/home/lxy/"+img_name+".jpg")
            print("id ",count)  
        elif count > 4002:
            break
        '''  
        for j in range(1,len(tem_str)):        
            shutil.copyfile("/data/common/HighRailway/photo/"+tem_str[j].strip(),"/data/common/forSZ/photo/"+tem_str[j].strip())
            print("photo ",j)
        '''

def gen_txt():
    txt_path = sys.argv[1]
    f_out = open(txt_path,'w')
    count = 0
    base_dir = "48/market_part/"
    for i in range(4000):
        path_ = base_dir + str(count)+".jpg"
        f_out.write("{} {}\n".format(path_,-1))
        count+=1

if __name__=='__main__':
    #cp_img()  
    gen_txt()   
