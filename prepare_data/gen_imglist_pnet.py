import numpy as np
import numpy.random as npr
import os
import argparse

def args():
    parser = argparse.ArgumentParser(description="generate Pnet train data")
    parser.add_argument('--data_dir',type=str,default="./12",\
                        help='annotation file saved dir')
    parser.add_argument('--out_dir',type=str,default="./imglists",\
                        help='generate train file saved dir')
    parser.add_argument('--size',type=int,default=12,\
                        help='select which net')
    parser.add_argument('--base_num',type=int,default=300000,\
                        help='part images annotion file ')
    parser.add_argument('--landmark',type=bool,default=False,\
                        help='if load landmark files ')
    parser.add_argument('--pos_txt',type=str,default="pos_12.txt",\
                        help='positive images annotion file ')
    parser.add_argument('--neg_txt',type=str,default="neg_12.txt",\
                        help='negtive images annotion file ')
    parser.add_argument('--part_txt',type=str,default="part_12.txt",\
                        help='part images annotion file ')
    return parser.parse_args()

def gen_p_imglist():
    parm = args()
    data_dir = parm.data_dir
    dir_path = parm.out_dir
    #anno_file = os.path.join(data_dir, "anno.txt")
    size = parm.size
    if size == 12:
        net = "PNet"
    elif size == 24:
        net = "RNet"
    elif size == 48:
        net = "ONet"
    '''
    with open(os.path.join(data_dir, 'pos_%s.txt' % (size)), 'r') as f:
        pos = f.readlines()

    with open(os.path.join(data_dir, 'neg_%s.txt' % (size)), 'r') as f:
        neg = f.readlines()

    with open(os.path.join(data_dir, 'part_%s.txt' % (size)), 'r') as f:
        part = f.readlines()

    with open(os.path.join(data_dir,'landmark_%s_aug.txt' %(size)), 'r') as f:
        landmark = f.readlines()
    '''
    with open(os.path.join(data_dir, parm.pos_txt), 'r') as f:
        pos = f.readlines()
    with open(os.path.join(data_dir, parm.neg_txt), 'r') as f:
        neg = f.readlines()
    with open(os.path.join(data_dir, parm.part_txt), 'r') as f:
        part = f.readlines()
    if parm.landmark:
        with open(os.path.join(data_dir,'landmark_%s_aug.txt' %(size)), 'r') as f:
            landmark = f.readlines()    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    save_dir = os.path.join(dir_path, "%s" %(net))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir,"train_%s_landmark.txt" % (net)), "w") as f:
        nums = [len(neg), len(pos), len(part)]
        ratio = [3, 1, 1]
        #base_num = min(nums)
        base_num = parm.base_num
        #base_num = 6*len(pos) - len(pos)-len(neg)-len(landmark)
        if parm.landmark:
            #print("neg,pos,part,landmark,base: ",len(neg), len(pos), len(part), len(landmark),base_num)
            #base_num = 6*len(pos) - len(pos)-len(part)-len(landmark)
            print("neg,pos,part,landmark,base: ",len(neg), len(pos), len(part), len(landmark),base_num)
        else:
            print("neg,pos,part,base: ",len(neg), len(pos), len(part),base_num)
        if len(neg) > 3*base_num :
            neg_keep = npr.choice(len(neg), size=3*base_num, replace=False)
        else:
            neg_keep = npr.choice(len(neg), size=3*base_num, replace=True)
        pos_keep = npr.choice(len(pos), size=base_num, replace=False)
        part_keep = npr.choice(len(part), size=base_num, replace=True)
        #pos_keep = npr.choice(len(pos), size=len(pos), replace=False)
        #part_keep = npr.choice(len(part), size=len(part), replace=False)
        print(len(neg_keep), len(pos_keep), len(part_keep))
        for i in pos_keep:
            f.write(pos[i])
        for i in neg_keep:
            f.write(neg[i])
        for i in part_keep:
            f.write(part[i])
        if parm.landmark:
            for item in landmark:
                f.write(item)
        f.close()

if __name__ == '__main__':
    gen_p_imglist()
