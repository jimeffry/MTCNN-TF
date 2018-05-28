#author: lxy
#time: 2018.3.19/19:00
#tool: python3
#version: om1
#modify:
######################################3
import numpy as np
import argparse
import csv
import os
import json

def args():
    parser = argparse.ArgumentParser(description='read csv')
    parser.add_argument('--in_file',type=str,default='/home/lxy/Downloads/DataSet/black.csv',\
                        help='csv file saved')
    parser.add_argument('--out_file',type=str,default=None,\
                        help='out file')
    return parser.parse_args()


def rd_csv(file_in,out_fil,pre_dir):
    f = open(file_in,'rb')
    cnt = 0
    tmp_dict = dict()
    for item_f in csv.DictReader(f):
        tmp_dict[item_f['filename']]=[]
        #print(item_f['filename'])
        #print(item_f['region_shape_attributes'])
        #new_dic = json.loads(item_f['region_shape_attributes'])
        #print(new_dic['x'])
    f.close()
    f = open(file_in,'rb')
    for item_f in csv.DictReader(f):
        new_dic = json.loads(item_f["region_shape_attributes"])
        #x,y,w,h
        a = [new_dic['x'],new_dic['y'],new_dic['x']+new_dic['width'],new_dic['y']+new_dic['height']]
        #print(item_f['filename'])
        tmp_dict[item_f['filename']].append(a)
    cnt=0
    for key_i in tmp_dict:
        print(key_i,tmp_dict[key_i])
        out_f.write("{}\t".format(pre_dir+key_i))
        tmp_arr = np.array(tmp_dict[key_i])
        row,col = np.shape(tmp_arr)
        for j in range(row):
            for k in range(col):
                out_f.write("{}\t".format(tmp_arr[j][k]))
        out_f.write("\n")
        cnt+=1
        #if cnt==5:
            #break
        print("total image ",cnt)
    f.close()
    
if __name__=='__main__':
    parm = args()
    f_in = parm.in_file
    f_list = ["/home/lxy/Downloads/DataSet/black.csv","/home/lxy/Downloads/DataSet/darklight2_b.csv","/home/lxy/Downloads/DataSet/overexposure2_b.csv"]
    pre_dir = ["overexposure1/","darklight2/","overexposure2/"]
    out_f = open("train_bright.txt",'w')
    for i in range(3):
        f_item = f_list[i]
        rd_csv(f_item,out_f,pre_dir[i])
    #rd_csv(f_in,out_f)
    out_f.close()