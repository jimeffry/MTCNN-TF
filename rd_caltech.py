#author : lxy
#time: 2018.3.23/ 11:30:00
#tool: python3
#version: 0.1
#modify:
#project: pedestrian detection
################################
import numpy as np 
import glob
import os
import argparse

def args():
    parser = argparse.ArgumentParser(description="read caltech txt")
    parser.add_argument('--dir_in',type=str,default="/home/lxy/Downloads/DataSet/trainval/",\
                        help='annotation files saved dir ')
    parser.add_argument('--out_file',type=str,default='train_caltech.txt',\
                        help='generated outfiles saved')
    return parser.parse_args()

def get_fil():
    parm = args()
    dir_in = parm.dir_in
    out_f = parm.out_file
    f_wt = open(out_f,'w')
    file_txts = glob.glob(dir_in+'annotations/*.txt')
    pass_cnt = 0
    for file_item in file_txts:
        f_rd = open(file_item,'r')
        line_list =  f_rd.readlines()
        if len(line_list)==0:
            f_rd.close()
            print("empyt file: ",file_item)
            pass_cnt+=1
            continue
        img_split = file_item.split('/')
        img_name = img_split[-1][:-4]
        img_lists = glob.glob(dir_in+'images/*')
        for img_one in img_lists:
            img_lists_split = img_one.split('/')
            img_one_name = img_lists_split[-1]
            if img_name in img_one_name:
                img_name = img_one_name
        f_wt.write("{} ".format(img_name))
        for line in line_list:
            line = line.strip()
            f_wt.write("{} ".format(line[1:]))
        f_wt.write("\n")
        f_rd.close()
    f_wt.close()
    print("pass ",pass_cnt)

if __name__=="__main__":
    get_fil()
